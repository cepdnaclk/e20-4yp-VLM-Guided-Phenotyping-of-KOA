"""
Fine-tuning BiomedCLIP for Knee OA — KL Grade Regression
==========================================================
Strategy  : Ordinal regression — predict KL grade (0-4) from knee image
Loss      : MSE + ordinal ranking loss
Frozen    : Text encoder + blocks[0-5] + patch_embed + pos_embed
Trainable : blocks[6-11] + trunk.norm + visual.head + regression head
Data      : All 8945 matched images (train/val 80/20 stratified)
Epochs    : 30 with early stopping (patience=6)
Batch     : 32
LR        : 3e-4 with cosine annealing

After training:
  - Regression head is discarded
  - Encoder produces knee-aware 512-dim embeddings
  - Re-run clustering + XAI with new embeddings

Usage:
    python finetune_regression.py

Outputs:
    /embeddings/finetuned_regression.pt          ← best checkpoint
    /embeddings/training_curves_regression.png   ← loss curves
    /embeddings/embeddings_V00_finetuned.npz     ← new embeddings
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import open_clip
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
NPZ_PATH   = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings/embeddings_V00.npz")
OUTPUT_DIR = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
DEVICE        = "cpu"
BATCH_SIZE    = 32
EPOCHS        = 30
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 6
IMAGE_SIZE    = 224
SEED          = 42
FREEZE_BLOCKS = list(range(6))   # freeze blocks 0-5, train 6-11
NUM_KL        = 5                # KL grades 0-4

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


# ── Loss: MSE + Ordinal Ranking ────────────────────────────────────────────────
class OrdinalRegressionLoss(nn.Module):
    """
    Combined loss for ordinal KL grade prediction:
    1. MSE loss — direct regression to KL grade value
    2. Ranking loss — ensure pred(KL=2) > pred(KL=1) > pred(KL=0) etc.

    The ranking loss reinforces the ordinal structure,
    which pure MSE alone doesn't guarantee.
    """
    def __init__(self, ranking_weight=0.3):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        predictions : (N,) predicted KL grades (float)
        targets     : (N,) true KL grades (float)
        """
        # MSE loss
        mse_loss = self.mse(predictions, targets)

        # Ranking loss — for all pairs (i,j) where target_i > target_j,
        # penalize if prediction_i <= prediction_j
        ranking_loss = torch.tensor(0.0)
        if len(predictions) > 1:
            # All pairs
            pred_i = predictions.unsqueeze(1)   # (N, 1)
            pred_j = predictions.unsqueeze(0)   # (1, N)
            tgt_i  = targets.unsqueeze(1)
            tgt_j  = targets.unsqueeze(0)

            # Mask: pairs where target_i > target_j
            pair_mask = (tgt_i - tgt_j) > 0.5  # at least 1 KL grade apart

            if pair_mask.sum() > 0:
                # Margin ranking: pred_i should be > pred_j when target_i > target_j
                margin      = 0.5
                rank_violations = torch.relu(margin - (pred_i - pred_j))
                ranking_loss = (rank_violations * pair_mask.float()).sum() / \
                               pair_mask.float().sum()

        total = mse_loss + self.ranking_weight * ranking_loss
        return total, mse_loss.item(), ranking_loss.item()


# ── Dataset ────────────────────────────────────────────────────────────────────
class KneeRegressionDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        img   = self.transform(img)
        label = torch.tensor(rec["kl_grade"], dtype=torch.float32)
        return img, label


def get_transforms(train=True):
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711)
    )
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


def build_records(npz_path, seed=42):
    """Load all matched images, stratified 80/20 train/val split."""
    d           = np.load(npz_path, allow_pickle=True)
    mask        = d["matched"]
    image_paths = d["image_paths"][mask]
    kl_grades   = d["kl_grades"][mask].astype(int)

    # Stratified split — preserve KL grade distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    indices = np.arange(len(image_paths))

    train_idx, val_idx = next(sss.split(indices, kl_grades))

    train_records = [
        {"image_path": str(image_paths[i]), "kl_grade": int(kl_grades[i])}
        for i in train_idx if Path(str(image_paths[i])).exists()
    ]
    val_records = [
        {"image_path": str(image_paths[i]), "kl_grade": int(kl_grades[i])}
        for i in val_idx if Path(str(image_paths[i])).exists()
    ]

    # Print distribution
    print(f"\nTrain: {len(train_records)} images")
    for kl in range(5):
        n = sum(1 for r in train_records if r["kl_grade"] == kl)
        print(f"  KL{kl}: {n}")

    print(f"\nVal: {len(val_records)} images")
    for kl in range(5):
        n = sum(1 for r in val_records if r["kl_grade"] == kl)
        print(f"  KL{kl}: {n}")

    return train_records, val_records


# ── Model ──────────────────────────────────────────────────────────────────────
class BiomedCLIPRegressor(nn.Module):
    """
    BiomedCLIP image encoder + regression head.
    The head is discarded after training — only encoder is used for embeddings.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual

        # Regression head: 512 → 256 → 1
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Get image features from encoder
        features = self.encoder(x)                    # (N, 512)
        features = F.normalize(features, dim=-1)
        pred     = self.head(features).squeeze(-1)    # (N,)
        return pred, features


def load_model():
    print("Loading BiomedCLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # Freeze text encoder
    for param in clip_model.text.parameters():
        param.requires_grad = False

    # Freeze visual: patch embed, pos embed, cls token
    clip_model.visual.trunk.cls_token.requires_grad = False
    clip_model.visual.trunk.pos_embed.requires_grad = False
    for param in clip_model.visual.trunk.patch_embed.parameters():
        param.requires_grad = False

    # Freeze transformer blocks 0-5
    for i in FREEZE_BLOCKS:
        for param in clip_model.visual.trunk.blocks[i].parameters():
            param.requires_grad = False

    # Build regressor
    model = BiomedCLIPRegressor(clip_model)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}  ({trainable/total*100:.1f}%)")
    print(f"  Frozen params   : {total-trainable:,}  ({(total-trainable)/total*100:.1f}%)")

    return model.to(DEVICE)


# ── Training ───────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_mse, total_rank = 0., 0., 0.
    all_preds, all_targets = [], []

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        preds, _ = model(images)
        loss, mse, rank = criterion(preds, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_mse  += mse
        total_rank += rank
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    n = len(loader)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    return total_loss/n, total_mse/n, total_rank/n, mae


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_mse = 0., 0.
    all_preds, all_targets = [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        preds, _ = model(images)
        loss, mse, _ = criterion(preds, labels)
        total_loss += loss.item()
        total_mse  += mse
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    n   = len(loader)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    return total_loss/n, total_mse/n, mae


def plot_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val")
    best_ep = np.argmin(history["val_loss"]) + 1
    axes[0].axvline(x=best_ep, color="green", linestyle="--", alpha=0.7,
                    label=f"Best epoch={best_ep}")
    axes[0].set_title("Total Loss (MSE + Ranking)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(epochs, history["train_mae"], "b-o", markersize=4, label="Train MAE")
    axes[1].plot(epochs, history["val_mae"],   "r-o", markersize=4, label="Val MAE")
    axes[1].set_title("Mean Absolute Error (KL grade units)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="MAE=0.5")

    # LR
    axes[2].plot(epochs, history["lr"], "g-o", markersize=4)
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves_regression.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curves saved → {path}")


# ── Re-extract embeddings ──────────────────────────────────────────────────────
def extract_embeddings(model, npz_path, output_dir):
    """Extract 512-dim embeddings from fine-tuned encoder (discard regression head)."""
    print("\nExtracting embeddings with fine-tuned encoder...")

    d           = np.load(npz_path, allow_pickle=True)
    mask        = d["matched"]
    image_paths = d["image_paths"][mask]

    transform = get_transforms(train=False)
    model.eval()

    all_emb  = []
    batch_paths = []

    def flush():
        if not batch_paths:
            return
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception:
                imgs.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
        imgs = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            _, features = model(imgs)
            features = features.float().cpu().numpy()
        all_emb.append(features)
        batch_paths.clear()

    for p in tqdm(image_paths, desc="  Extracting"):
        batch_paths.append(str(p))
        if len(batch_paths) == 32:
            flush()
    flush()

    embeddings = np.vstack(all_emb)
    print(f"  Shape: {embeddings.shape}")

    out_path = output_dir / "embeddings_V00_finetuned.npz"
    np.savez(
        out_path,
        embeddings  = embeddings,
        image_paths = image_paths,
        subject_ids = d["subject_ids"][mask],
        knee_sides  = d["knee_sides"][mask],
        kl_grades   = d["kl_grades"][mask],
        pain_scores = d["pain_scores"][mask],
        bmis        = d["bmis"][mask],
        ages        = d["ages"][mask],
        jsn_lateral = d["jsn_lateral"][mask],
        jsn_medial  = d["jsn_medial"][mask],
        matched     = np.ones(embeddings.shape[0], dtype=bool),
    )
    print(f"  Saved → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BiomedCLIP Fine-tuning — KL Grade Regression")
    print("=" * 60)

    # Dataset
    train_records, val_records = build_records(NPZ_PATH)

    train_loader = DataLoader(
        KneeRegressionDataset(train_records, get_transforms(train=True)),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        KneeRegressionDataset(val_records, get_transforms(train=False)),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    print(f"\nBatches per epoch — train: {len(train_loader)}  val: {len(val_loader)}")

    # Model
    model     = load_model()
    criterion = OrdinalRegressionLoss(ranking_weight=0.3)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Training
    print(f"\nTraining — {EPOCHS} epochs, early stopping patience={PATIENCE}")
    print("=" * 60)

    history = {
        "train_loss": [], "val_loss": [],
        "train_mae":  [], "val_mae":  [], "lr": []
    }
    best_val     = float("inf")
    patience_ctr = 0
    best_ckpt    = OUTPUT_DIR / "finetuned_regression.pt"

    for epoch in range(1, EPOCHS + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:02d}/{EPOCHS}  lr={lr_now:.2e}")

        tr_loss, tr_mse, tr_rank, tr_mae = train_epoch(
            model, train_loader, optimizer, criterion
        )
        vl_loss, vl_mse, vl_mae = val_epoch(
            model, val_loader, criterion
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)
        history["lr"].append(lr_now)

        print(f"  train_loss={tr_loss:.4f}  (mse={tr_mse:.4f} rank={tr_rank:.4f})  mae={tr_mae:.3f}")
        print(f"  val_loss  ={vl_loss:.4f}  (mse={vl_mse:.4f})  mae={vl_mae:.3f}")

        if vl_loss < best_val:
            best_val     = vl_loss
            patience_ctr = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    vl_loss,
                "val_mae":     vl_mae,
            }, best_ckpt)
            print(f"  ✓ Best model saved  (val_mae={vl_mae:.3f} KL grades)")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Plot
    plot_curves(history, OUTPUT_DIR)

    # Load best and extract embeddings
    best_epoch = np.argmin(history["val_loss"]) + 1
    print(f"\nLoading best model from epoch {best_epoch}...")
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    emb_path = extract_embeddings(model, NPZ_PATH, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Best val MAE  : {ckpt['val_mae']:.3f} KL grade units")
    print(f"Best epoch    : {best_epoch}")
    print(f"Embeddings    : {emb_path}")
    print(f"\nNext steps:")
    print(f"  1. Re-run clustering.py → point NPZ_PATH to embeddings_V00_finetuned.npz")
    print(f"  2. Re-run xai_visualization.py → verify attention on joint space")


if __name__ == "__main__":
    main()