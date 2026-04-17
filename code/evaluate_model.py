"""
Model Accuracy Evaluation
==========================
Evaluates the fine-tuned BiomedCLIP regression model on the validation set.

Metrics:
  - MAE, RMSE, R2 (regression)
  - Exact accuracy, Within-1-grade accuracy (classification)
  - Per-class precision, recall, F1
  - Per-KL grade MAE
  - Confusion matrix (counts + normalized)

Usage:
    python evaluate_model.py
"""

import torch
import numpy as np
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
NPZ_PATH = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings/embeddings_V00.npz")
CKPT     = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings/finetuned_regression.pt")
OUT_DIR  = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings")

DEVICE     = "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED       = 42


# ── Model definition ───────────────────────────────────────────────────────────
class BiomedCLIPRegressor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = F.normalize(features, dim=-1)
        return self.head(features).squeeze(-1), features


def load_model():
    print("Loading fine-tuned BiomedCLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = BiomedCLIPRegressor(clip_model)
    ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}  val_mae={ckpt['val_mae']:.3f}")
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std =(0.26862954, 0.26130258, 0.27577711)
        ),
    ])


def run_evaluation(model, image_paths, kl_grades, val_idx, transform):
    print(f"\nEvaluating on {len(val_idx)} validation images...")

    all_preds, all_true = [], []
    batch_imgs, batch_labels = [], []

    def flush():
        if not batch_imgs:
            return
        imgs = torch.stack(batch_imgs)
        with torch.no_grad():
            preds, _ = model(imgs)
        all_preds.extend(preds.float().numpy())
        all_true.extend(batch_labels)
        batch_imgs.clear()
        batch_labels.clear()

    for count, i in enumerate(val_idx):
        try:
            img = Image.open(str(image_paths[i])).convert("RGB")
            batch_imgs.append(transform(img))
            batch_labels.append(int(kl_grades[i]))
            if len(batch_imgs) == BATCH_SIZE:
                flush()
        except Exception as e:
            print(f"  Skipped {image_paths[i]}: {e}")

        if (count + 1) % 200 == 0:
            print(f"  {count+1}/{len(val_idx)} done...")

    flush()
    return np.array(all_preds), np.array(all_true)


def print_metrics(preds_arr, true_arr):
    # ── Regression ─────────────────────────────────────────────────────────
    mae  = np.mean(np.abs(preds_arr - true_arr))
    rmse = np.sqrt(np.mean((preds_arr - true_arr) ** 2))
    r2   = 1 - (np.sum((preds_arr - true_arr)**2) /
                np.sum((true_arr - true_arr.mean())**2))

    print(f"\n{'='*45}")
    print(f"REGRESSION METRICS  (n={len(true_arr)})")
    print(f"{'='*45}")
    print(f"  MAE        = {mae:.3f}  KL grade units")
    print(f"  RMSE       = {rmse:.3f}  KL grade units")
    print(f"  R2 score   = {r2:.3f}")

    # ── Classification ─────────────────────────────────────────────────────
    preds_cls = np.clip(np.round(preds_arr).astype(int), 0, 4)
    exact_acc = (preds_cls == true_arr).mean() * 100
    within1   = (np.abs(preds_cls - true_arr) <= 1).mean() * 100

    print(f"\n{'='*45}")
    print(f"CLASSIFICATION METRICS  (rounded predictions)")
    print(f"{'='*45}")
    print(f"  Exact accuracy          = {exact_acc:.1f}%")
    print(f"  Within-1-grade accuracy = {within1:.1f}%")

    print(f"\nPer-class Report:")
    print(classification_report(
        true_arr, preds_cls,
        target_names=["KL0", "KL1", "KL2", "KL3", "KL4"]
    ))

    print(f"Per-KL Grade MAE:")
    for kl in range(5):
        idx = true_arr == kl
        if idx.sum() > 0:
            kl_mae  = np.mean(np.abs(preds_arr[idx] - true_arr[idx]))
            kl_mean = preds_arr[idx].mean()
            print(f"  KL{kl}  (n={idx.sum():4d}):  "
                  f"MAE={kl_mae:.3f}  mean_pred={kl_mean:.2f}")

    return preds_cls, mae, rmse, exact_acc, within1


def save_confusion_matrix(true_arr, preds_cls, mae, exact_acc, within1, out_dir):
    cm      = confusion_matrix(true_arr, preds_cls)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["KL0","KL1","KL2","KL3","KL4"],
                yticklabels=["KL0","KL1","KL2","KL3","KL4"])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True KL Grade")
    axes[0].set_xlabel("Predicted KL Grade")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=["KL0","KL1","KL2","KL3","KL4"],
                yticklabels=["KL0","KL1","KL2","KL3","KL4"])
    axes[1].set_title("Confusion Matrix (normalized per true class)")
    axes[1].set_ylabel("True KL Grade")
    axes[1].set_xlabel("Predicted KL Grade")

    plt.suptitle(
        f"KL Grade Prediction  —  "
        f"MAE={mae:.3f}  |  Exact={exact_acc:.1f}%  |  Within-1={within1:.1f}%",
        fontsize=11
    )
    plt.tight_layout()
    out_path = out_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved → {out_path}")


def save_prediction_distribution(preds_arr, true_arr, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        true_arr + np.random.normal(0, 0.05, len(true_arr)),
        preds_arr, alpha=0.3, s=5, color="steelblue"
    )
    axes[0].plot([0, 4], [0, 4], "r--", linewidth=2, label="Perfect prediction")
    axes[0].set_xlabel("True KL Grade")
    axes[0].set_ylabel("Predicted KL Grade")
    axes[0].set_title("True vs Predicted KL Grade")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
    for kl in range(5):
        idx = true_arr == kl
        if idx.sum() > 0:
            axes[1].hist(preds_arr[idx], bins=20, alpha=0.6,
                         label=f"KL{kl} (n={idx.sum()})",
                         color=colors[kl])
    axes[1].set_xlabel("Predicted KL Grade")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction Distribution per True KL Grade")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "prediction_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction distribution saved → {out_path}")


def main():
    print("=" * 50)
    print("BiomedCLIP Regression — Model Evaluation")
    print("=" * 50)

    model     = load_model()
    transform = get_transform()

    d           = np.load(NPZ_PATH, allow_pickle=True)
    mask        = d["matched"]
    image_paths = d["image_paths"][mask]
    kl_grades   = d["kl_grades"][mask].astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    _, val_idx = next(sss.split(np.arange(len(image_paths)), kl_grades))

    preds_arr, true_arr = run_evaluation(
        model, image_paths, kl_grades, val_idx, transform
    )

    preds_cls, mae, rmse, exact_acc, within1 = print_metrics(
        preds_arr, true_arr
    )

    save_confusion_matrix(true_arr, preds_cls, mae, exact_acc, within1, OUT_DIR)
    save_prediction_distribution(preds_arr, true_arr, OUT_DIR)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"\nCopy results to local:")
    print(f"  scp e20378@ada:{OUT_DIR}/confusion_matrix.png ./")
    print(f"  scp e20378@ada:{OUT_DIR}/prediction_distribution.png ./")


if __name__ == "__main__":
    main()