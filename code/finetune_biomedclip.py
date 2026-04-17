"""
Fine-tuning BiomedCLIP for Knee OA Phenotyping
================================================
Strategy  : Supervised Contrastive Learning (SupCon)
Supervision: KL grade (0-4) as weak label
Frozen    : Text encoder + blocks[0-5] + patch_embed + pos_embed
Trainable : blocks[6-11] + trunk.norm + visual.head
Data      : Balanced 300 per KL grade (1500 train + 375 val)
Loss      : SupConLoss (temperature=0.07)
Epochs    : 30 with early stopping (patience=5)
Batch     : 64
LR        : 1e-5 with cosine annealing

Usage:
    pip install open_clip_torch pandas tqdm matplotlib
    python finetune_biomedclip.py

Outputs:
    /embeddings/finetuned_model.pt        ← best checkpoint
    /embeddings/finetuned_last.pt         ← last epoch
    /embeddings/training_curves.png       ← loss + lr curves
    /embeddings/embeddings_V00_finetuned.npz ← new embeddings
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import open_clip
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
CROP_DIR    = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/cropped_yolo_V00_images")
META_CSV    = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/master_final_V00.csv")
NPZ_PATH    = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings/embeddings_V00.npz")
OUTPUT_DIR  = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
DEVICE          = "cpu"
SAMPLES_PER_KL  = 600       # balanced: 300 per grade × 5 grades = 1500 train
VAL_PER_KL      = 100        # 75 per grade × 5 = 375 val
BATCH_SIZE      = 32
EPOCHS          = 20
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
TEMPERATURE     = 0.1      # SupCon temperature
PATIENCE        = 6         # early stopping patience
IMAGE_SIZE      = 224
SEED            = 42
FREEZE_BLOCKS   = list(range(6))   # freeze blocks 0-5, train 6-11

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")


# ── Supervised Contrastive Loss ────────────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls embeddings of same KL grade together,
    pushes embeddings of different KL grades apart.

    Args:
        temperature: softmax temperature (lower = sharper separation)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features : (N, D) L2-normalized embeddings
        labels   : (N,)   KL grade labels (0-4)
        """
        device = features.device
        N = features.shape[0]

        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # Mask: same label pairs (excluding diagonal)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exp similarities (exclude diagonal via mask)
        exp_sim = torch.exp(sim)
        diag_mask = (~torch.eye(N, dtype=torch.bool, device=device)).float()
        exp_sim = exp_sim * diag_mask

        # Log probability
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs per anchor
        n_pos = pos_mask.sum(dim=1)
        loss  = -(pos_mask * log_prob).sum(dim=1)

        # Only compute loss for anchors that have at least one positive
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = (loss[valid] / n_pos[valid]).mean()
        return loss


# ── Dataset ────────────────────────────────────────────────────────────────────
class KneeDataset(Dataset):
    """
    Loads knee crops with KL grade labels.
    Applies augmentation for training, clean preprocessing for validation.
    """
    def __init__(self, records, transform):
        """
        records: list of dicts with keys: image_path, kl_grade
        """
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        img      = Image.open(rec["image_path"]).convert("RGB")
        img      = self.transform(img)
        label    = torch.tensor(rec["kl_grade"], dtype=torch.long)
        return img, label


def get_transforms(train=True):
    """
    Training: augmentation to improve generalization.
    Validation: clean resize + normalize only.
    """
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711)
    )
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


def build_dataset_records(npz_path, crop_dir, samples_per_kl, val_per_kl, seed=42):
    """
    Load matched image paths + KL grades from npz.
    Split into balanced train/val sets.
    """
    d           = np.load(npz_path, allow_pickle=True)
    mask        = d["matched"]
    image_paths = d["image_paths"][mask]
    kl_grades   = d["kl_grades"][mask].astype(int)

    rng = np.random.default_rng(seed)

    train_records, val_records = [], []

    for kl in range(5):
        indices = np.where(kl_grades == kl)[0]
        n_available = len(indices)
        n_needed    = samples_per_kl + val_per_kl

        if n_available < n_needed:
            print(f"  KL{kl}: only {n_available} available, using all "
                  f"({samples_per_kl} train + min({val_per_kl}, rest) val)")
            chosen = indices
        else:
            chosen = rng.choice(indices, size=n_needed, replace=False)

        n_train = min(samples_per_kl, len(chosen) - 1)
        n_val   = len(chosen) - n_train

        train_idx = chosen[:n_train]
        val_idx   = chosen[n_train:]

        for i in train_idx:
            path = str(image_paths[i])
            if Path(path).exists():
                train_records.append({"image_path": path, "kl_grade": kl})

        for i in val_idx:
            path = str(image_paths[i])
            if Path(path).exists():
                val_records.append({"image_path": path, "kl_grade": kl})

        print(f"  KL{kl}: {len(train_idx)} train | {len(val_idx)} val")

    print(f"\nTotal: {len(train_records)} train | {len(val_records)} val")
    return train_records, val_records


# ── Model setup ────────────────────────────────────────────────────────────────
def load_and_freeze_model():
    """
    Load BiomedCLIP and freeze:
      - Entire text encoder
      - patch_embed, pos_embed, cls_token
      - transformer blocks 0-5

    Train:
      - transformer blocks 6-11
      - trunk.norm
      - visual.head
    """
    print("Loading BiomedCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # ── Freeze text encoder completely ────────────────────────────────────────
    for param in model.text.parameters():
        param.requires_grad = False

    # ── Freeze visual: patch embed, positional embed, cls token ──────────────
    model.visual.trunk.cls_token.requires_grad  = False
    model.visual.trunk.pos_embed.requires_grad  = False
    for param in model.visual.trunk.patch_embed.parameters():
        param.requires_grad = False

    # ── Freeze transformer blocks 0-5 ─────────────────────────────────────────
    for i in FREEZE_BLOCKS:
        for param in model.visual.trunk.blocks[i].parameters():
            param.requires_grad = False

    # ── Blocks 6-11, trunk.norm, visual.head stay trainable ──────────────────
    # (they're already requires_grad=True by default)

    # Summary
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}  ({trainable/total*100:.1f}%)")
    print(f"  Frozen params   : {frozen:,}  ({frozen/total*100:.1f}%)")

    return model.to(DEVICE), preprocess


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
            features = model.encode_image(images)
            features = F.normalize(features, dim=-1)
            loss     = criterion(features, labels)

        if DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
            features = model.encode_image(images)
            features = F.normalize(features, dim=-1)
            loss     = criterion(features, labels)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def plot_curves(train_losses, val_losses, lr_history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train loss")
    ax1.plot(epochs, val_losses,   "r-o", markersize=4, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SupCon Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val   = min(val_losses)
    ax1.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                label=f"Best epoch={best_epoch} (val={best_val:.4f})")
    ax1.legend()

    ax2.plot(epochs, lr_history, "g-o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {path}")


# ── Re-extract embeddings with fine-tuned model ───────────────────────────────
def extract_finetuned_embeddings(model, preprocess, npz_path, output_dir):
    """Re-extract all 8945 embeddings using the fine-tuned model."""
    print("\nRe-extracting embeddings with fine-tuned model...")

    d           = np.load(npz_path, allow_pickle=True)
    mask        = d["matched"]
    image_paths = d["image_paths"][mask]
    all_paths   = d["image_paths"]

    model.eval()
    val_transform = get_transforms(train=False)

    all_embeddings = []
    batch_paths    = []
    BATCH          = 64

    def flush_batch():
        if not batch_paths:
            return
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(val_transform(img))
            except Exception:
                imgs.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
        imgs = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
                emb = model.encode_image(imgs)
                emb = F.normalize(emb, dim=-1)
        all_embeddings.append(emb.cpu().numpy())
        batch_paths.clear()

    for path in tqdm(image_paths, desc="  Extracting"):
        batch_paths.append(path)
        if len(batch_paths) == BATCH:
            flush_batch()
    flush_batch()

    embeddings = np.vstack(all_embeddings)
    print(f"  Extracted: {embeddings.shape}")

    # Build new npz reusing all metadata from original
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
    print(f"Finetuned embeddings saved → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BiomedCLIP Fine-tuning — Knee OA Supervised Contrastive")
    print("=" * 60)

    # Build dataset
    print("\nBuilding balanced dataset...")
    train_records, val_records = build_dataset_records(
        NPZ_PATH, CROP_DIR, SAMPLES_PER_KL, VAL_PER_KL
    )

    train_dataset = KneeDataset(train_records, get_transforms(train=True))
    val_dataset   = KneeDataset(val_records,   get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False
    )

    # Load model
    print("\nSetting up model...")
    model, preprocess = load_and_freeze_model()

    # Loss + optimizer + scheduler
    criterion = SupConLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    # Training loop
    print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={PATIENCE})")
    print("=" * 60)

    train_losses = []
    val_losses   = []
    lr_history   = []
    best_val     = float("inf")
    patience_ctr = 0
    best_ckpt    = OUTPUT_DIR / "finetuned_model.pt"

    for epoch in range(1, EPOCHS + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:02d}/{EPOCHS}  lr={lr_now:.2e}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss   = val_epoch(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(lr_now)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val     = val_loss
            patience_ctr = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "train_loss":  train_loss,
            }, best_ckpt)
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save last checkpoint
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_loss":    val_losses[-1],
    }, OUTPUT_DIR / "finetuned_last.pt")

    # Plot curves
    plot_curves(train_losses, val_losses, lr_history, OUTPUT_DIR)

    # Load best model and re-extract embeddings
    print(f"\nLoading best model from epoch {np.argmin(val_losses)+1}...")
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    emb_path = extract_finetuned_embeddings(model, preprocess, NPZ_PATH, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Best val loss     : {best_val:.4f}")
    print(f"Best checkpoint   : {best_ckpt}")
    print(f"Finetuned embeddings: {emb_path}")
    print("\nNext step: re-run clustering.py pointing to embeddings_V00_finetuned.npz")
    print("Then re-run xai_visualization.py to see improved attention maps")


if __name__ == "__main__":
    main()