"""
XAI Visualization — Last Layer CLS Attention
=============================================
Uses the last transformer block's CLS token attention to visualize
which regions of the knee the fine-tuned model focuses on.

This is the correct XAI method for ViT models (not GradCAM++, not rollout).
The CLS token aggregates global image information — its attention to each
patch in the last layer shows what the model considered most important
for its final decision.

Outputs per KL grade:
  - KL{n}_cluster{c}_attention.png  : 3 sample images per cluster
  - KL{n}_comparison.png            : all clusters side by side for comparison

Usage:
    python xai_cls_attention.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import pandas as pd
from pathlib import Path
import open_clip
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT        = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings/finetuned_regression.pt")
CLUSTER_CSV = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/clustering/cluster_assignments_finetuned.csv")
OUTPUT_DIR  = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/clustering/xai_cls")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE          = "cpu"
IMAGE_SIZE      = 224
SAMPLES_PER_CLUSTER = 3
SEED            = 42

# ── Model ──────────────────────────────────────────────────────────────────────
class BiomedCLIPRegressor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 1)
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


# ── CLS Attention Extractor ────────────────────────────────────────────────────
class CLSAttentionExtractor:
    """
    Extracts CLS token attention from the last transformer block.
    The CLS token's attention weights to each patch show which
    regions the model focuses on for its global representation.
    """
    def __init__(self, model):
        self.model    = model
        self.last_attn = None
        self._hook    = model.encoder.trunk.blocks[-1].attn.register_forward_hook(
            self._hook_fn
        )

    def _hook_fn(self, module, input, output):
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            scale = (C // module.num_heads) ** -0.5
            attn  = (q @ k.transpose(-2, -1)) * scale
            attn  = attn.softmax(dim=-1)
            # Average over heads, take CLS→patch attention
            self.last_attn = attn.mean(dim=1)[:, 0, 1:].detach()  # (B, 196)

    def get_attention_map(self, img_tensor):
        """Returns normalized 14×14 attention map."""
        with torch.no_grad():
            self.model(img_tensor.to(DEVICE))

        attn = self.last_attn[0].cpu().numpy()  # (196,)
        attn = attn.reshape(14, 14)

        # Normalize to 0-1
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        return attn

    def remove_hook(self):
        self._hook.remove()


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std =(0.26862954, 0.26130258, 0.27577711)
        ),
    ])


def overlay_attention(img_pil, attn_map, alpha=0.45, colormap=cv2.COLORMAP_INFERNO):
    """Overlay attention heatmap on image."""
    img_np      = np.array(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    attn_resized = cv2.resize(attn_map, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap     = cv2.applyColorMap(np.uint8(255 * attn_resized), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = np.clip((1-alpha)*img_np + alpha*heatmap_rgb, 0, 255).astype(np.uint8)
    return overlay


def find_image(row, crop_dir=Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/cropped_yolo_V00_images")):
    """Find image path from cluster CSV row."""
    if "image_path" in row and pd.notna(row["image_path"]) and Path(str(row["image_path"])).exists():
        return Path(str(row["image_path"]))
    knee_num = "knee1" if str(row["knee_side"]) == "LEFT" else "knee0"
    matches  = list(crop_dir.glob(f"*{row['subject_id']}*{knee_num}*.png"))
    return matches[0] if matches else None


def get_predicted_kl(model, img_tensor):
    """Get model's predicted KL grade for an image."""
    with torch.no_grad():
        pred, _ = model(img_tensor.to(DEVICE))
    return float(pred.item())


# ── Per-cluster visualization ──────────────────────────────────────────────────
def visualize_cluster(kl, cluster_id, c_df, model, extractor, transform, output_dir):
    """Generate attention visualization for one cluster."""
    n_total  = len(c_df)
    samples  = c_df.sample(min(SAMPLES_PER_CLUSTER, n_total), random_state=SEED)

    # Cluster metadata summary
    pain_mean   = c_df["pain"].mean()
    jsn_lat     = c_df["jsn_lat"].mean()
    jsn_med     = c_df["jsn_med"].mean()
    bmi_mean    = c_df["bmi"].mean()
    age_mean    = c_df["age"].mean()

    fig, axes = plt.subplots(
        len(samples), 2,
        figsize=(10, 4.5 * len(samples))
    )
    if len(samples) == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"KL Grade {kl}  —  Cluster {cluster_id}  (n={n_total})\n"
        f"pain={pain_mean:.1f}  jsn_lat={jsn_lat:.2f}  jsn_med={jsn_med:.2f}"
        f"  bmi={bmi_mean:.1f}  age={age_mean:.0f}y",
        fontsize=11, y=1.01
    )

    valid_count = 0
    for row_idx, (_, sample) in enumerate(samples.iterrows()):
        img_path = find_image(sample)
        if img_path is None:
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")
            continue

        img_pil    = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0)

        # Get attention map
        attn_map = extractor.get_attention_map(img_tensor)
        overlay  = overlay_attention(img_pil, attn_map)

        # Get model's KL prediction for this image
        pred_kl = get_predicted_kl(model, img_tensor)

        # Original
        axes[row_idx, 0].imshow(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[row_idx, 0].set_title(
            f"pain={int(sample['pain'])}  "
            f"jsn_m={int(sample['jsn_med'])}  "
            f"jsn_l={int(sample['jsn_lat'])}  "
            f"pred_KL={pred_kl:.1f}",
            fontsize=8
        )
        axes[row_idx, 0].axis("off")

        # Attention overlay
        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].set_title("CLS Attention (last block)", fontsize=8)
        axes[row_idx, 1].axis("off")

        valid_count += 1

    plt.tight_layout()
    out_path = output_dir / f"KL{kl}_cluster{cluster_id}_attention.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── Cross-cluster comparison plot ─────────────────────────────────────────────
def comparison_plot(kl, kl_df, model, extractor, transform, output_dir):
    """
    One figure showing one representative sample per cluster side by side.
    Makes it easy to compare attention patterns across phenotypes.
    """
    clusters = sorted([c for c in kl_df["cluster"].unique() if c != -1])
    n_clusters = len(clusters)

    if n_clusters == 0:
        return

    fig, axes = plt.subplots(
        2, n_clusters,
        figsize=(5.5 * n_clusters, 11)
    )
    if n_clusters == 1:
        axes = axes[:, None]

    fig.suptitle(
        f"KL Grade {kl} — Phenotype Comparison\n"
        f"Top: Original X-ray   Bottom: CLS Attention",
        fontsize=13, y=1.02
    )

    for col, cluster_id in enumerate(clusters):
        c_df     = kl_df[kl_df["cluster"] == cluster_id]
        n_total  = len(c_df)
        pain_m   = c_df["pain"].mean()
        jsn_lat  = c_df["jsn_lat"].mean()
        jsn_med  = c_df["jsn_med"].mean()

        # Pick the sample closest to cluster mean pain
        pain_diff = (c_df["pain"] - pain_m).abs()
        rep_row   = c_df.iloc[pain_diff.argmin()]
        img_path  = find_image(rep_row)

        if img_path is None:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        img_pil    = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0)
        attn_map   = extractor.get_attention_map(img_tensor)
        overlay    = overlay_attention(img_pil, attn_map)
        pred_kl    = get_predicted_kl(model, img_tensor)

        # Determine phenotype label
        if jsn_lat > jsn_med:
            phenotype = "Lateral JSN"
            color     = "#e74c3c"
        elif jsn_med > jsn_lat:
            phenotype = "Medial JSN"
            color     = "#2980b9"
        else:
            phenotype = "No JSN"
            color     = "#27ae60"

        # Column title
        col_title = (
            f"Cluster {cluster_id}: {phenotype}\n"
            f"n={n_total}  pain={pain_m:.1f}\n"
            f"jsn_lat={jsn_lat:.2f}  jsn_med={jsn_med:.2f}"
        )
        axes[0, col].set_title(col_title, fontsize=9, color=color, fontweight="bold")

        axes[0, col].imshow(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"pred_KL={pred_kl:.1f}", fontsize=8)
        axes[1, col].axis("off")

        # Add colored border to bottom of image column
        for spine in axes[0, col].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        for spine in axes[1, col].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    out_path = output_dir / f"KL{kl}_phenotype_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot → {out_path.name}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("XAI — Last Layer CLS Attention Visualization")
    print("=" * 60)

    # Load
    cluster_df = pd.read_csv(CLUSTER_CSV)
    print(f"Loaded {len(cluster_df)} cluster assignments")

    model     = load_model()
    extractor = CLSAttentionExtractor(model)
    transform = get_transform()

    # Run for KL grades 0-4
    for kl in range(5):
        kl_df    = cluster_df[cluster_df["kl_grade"] == kl]
        clusters = sorted([c for c in kl_df["cluster"].unique() if c != -1])

        print(f"\n── KL Grade {kl}  ({len(kl_df)} knees, {len(clusters)} clusters) ──")

        # Individual cluster plots
        for cluster_id in clusters:
            c_df = kl_df[kl_df["cluster"] == cluster_id]
            out  = visualize_cluster(
                kl, cluster_id, c_df, model, extractor, transform, OUTPUT_DIR
            )
            print(f"  Cluster {cluster_id} (n={len(c_df)}) → {out.name}")

        # Cross-cluster comparison
        comparison_plot(kl, kl_df, model, extractor, transform, OUTPUT_DIR)

    extractor.remove_hook()

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print(f"\nCopy to local:")
    print(f"  scp -r e20378@ada:{OUTPUT_DIR} ./xai_cls_results")


if __name__ == "__main__":
    main()