"""
XAI Visualization for BiomedCLIP Knee OA Clusters
===================================================
Runs both GradCAM++ and Attention Rollout on sample images
from each cluster within each KL grade.

Output: /clustering/xai/
    KL{n}_cluster{c}_sample{i}_gradcam.png
    KL{n}_cluster{c}_sample{i}_rollout.png
    KL{n}_cluster{c}_comparison.png   <- side by side both methods

Usage:
    pip install grad-cam
    python xai_visualization.py
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import pandas as pd
from pathlib import Path
import open_clip
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
NPZ_PATH     = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings/embeddings_V00_finetuned.npz")
CLUSTER_CSV  = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/clustering/cluster_assignments_finetuned.csv")
OUTPUT_DIR   = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/clustering/xai_finetuned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
SAMPLES_PER_CLUSTER = 3   # how many images to visualize per cluster
IMAGE_SIZE   = 224        # BiomedCLIP input size


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading BiomedCLIP on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.eval().to(DEVICE)
    print("Model loaded.")
    return model, preprocess


# ── Attention Rollout ─────────────────────────────────────────────────────────
class AttentionRollout:
    """
    Computes attention rollout for ViT models.
    Multiplies attention matrices across all transformer layers
    to find which image patches the model attended to most.
    """
    def __init__(self, model, discard_ratio=0.9):
        self.model        = model
        self.discard_ratio = discard_ratio
        self.attention_maps = []
        self.hooks          = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into each transformer block's attention module."""
        visual = self.model.visual.trunk  # ViT backbone

        for block in visual.blocks:
            hook = block.attn.register_forward_hook(self._attention_hook)
            self.hooks.append(hook)

    def _attention_hook(self, module, input, output):
        """Capture attention weights during forward pass."""
        # Get attention weights from the module
        # For timm ViT, we need to recompute attention weights
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3,
                  module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            scale    = (C // module.num_heads) ** -0.5
            attn     = (q @ k.transpose(-2, -1)) * scale
            attn     = attn.softmax(dim=-1)
            # Average over heads
            attn_avg = attn.mean(dim=1)  # (B, N, N)
            self.attention_maps.append(attn_avg.detach().cpu())

    def __call__(self, img_tensor):
        self.attention_maps = []

        with torch.no_grad():
            _ = self.model.encode_image(img_tensor.to(DEVICE))

        # Rollout: multiply attention matrices layer by layer
        # Add identity matrix (residual connection)
        result = torch.eye(self.attention_maps[0].shape[-1])

        for attn in self.attention_maps:
            attn_map = attn[0]  # single image

            # Add identity (residual)
            attn_with_residual = attn_map + torch.eye(attn_map.shape[-1])
            attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

            # Discard lowest attention values (noise reduction)
            flat     = attn_with_residual.view(-1)
            thresh   = torch.quantile(flat, self.discard_ratio)
            attn_with_residual[attn_with_residual < thresh] = 0

            result = torch.matmul(attn_with_residual, result)

        # Extract CLS token attention to all patches
        # CLS token is index 0, patches are 1:
        mask = result[0, 1:]   # shape: (196,) for 14×14 patches

        # Reshape to 2D patch grid
        num_patches = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(num_patches, num_patches).numpy()

        # Normalize to 0-1
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


# ── GradCAM++ ─────────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    """
    GradCAM++ adapted for ViT.
    Applies to the last transformer block's output projection.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        visual = self.model.visual.trunk
        # Target: last transformer block
        target_layer = visual.blocks[-1].mlp

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(
            target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            target_layer.register_full_backward_hook(backward_hook)
        )

    def __call__(self, img_tensor):
        self.model.zero_grad()
        img_tensor = img_tensor.to(DEVICE).requires_grad_(True)

        # Forward pass — get image features
        features = self.model.encode_image(img_tensor)

        # Use L2 norm of features as scalar target
        score = features.norm()
        score.backward()

        if self.gradients is None or self.activations is None:
            print("  Warning: GradCAM++ hooks did not capture gradients")
            return np.zeros((14, 14))

        grads  = self.gradients[0]       # (N_tokens, C)
        acts   = self.activations[0]     # (N_tokens, C)

        # GradCAM++ weights
        grads_sq  = grads ** 2
        grads_cub = grads ** 3
        denom     = 2 * grads_sq + acts.sum(dim=0, keepdim=True) * grads_cub
        denom     = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha     = grads_sq / denom
        weights   = (alpha * torch.relu(grads)).sum(dim=-1)  # (N_tokens,)

        # Remove CLS token, reshape to patch grid
        patch_weights = weights[1:]   # (196,)
        num_patches   = int(patch_weights.shape[0] ** 0.5)
        cam = patch_weights.reshape(num_patches, num_patches).cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


# ── Overlay heatmap on image ──────────────────────────────────────────────────
def overlay_heatmap(img_pil, cam_map, colormap=cv2.COLORMAP_JET, alpha=0.45):
    """Resize CAM to image size and overlay as heatmap."""
    img_np  = np.array(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))

    # Resize cam to image size
    cam_resized = cv2.resize(cam_map, (IMAGE_SIZE, IMAGE_SIZE))
    cam_uint8   = np.uint8(255 * cam_resized)
    heatmap     = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (1 - alpha) * img_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ── Main visualization ────────────────────────────────────────────────────────
def visualize_kl_grade(kl, cluster_df, model, preprocess,
                        rollout, gradcam):
    """Generate XAI visualizations for all clusters in one KL grade."""

    kl_df      = cluster_df[cluster_df["kl_grade"] == kl]
    clusters   = sorted(kl_df["cluster"].unique())
    clusters   = [c for c in clusters if c != -1]  # skip noise

    print(f"\n── KL Grade {kl} — {len(clusters)} clusters ──")

    for cluster_id in clusters:
        c_df    = kl_df[kl_df["cluster"] == cluster_id]
        samples = c_df.sample(min(SAMPLES_PER_CLUSTER, len(c_df)),
                               random_state=42)

        print(f"  Cluster {cluster_id}: {len(c_df)} knees, "
              f"visualizing {len(samples)} samples")

        fig, axes = plt.subplots(
            len(samples), 3,
            figsize=(12, 4 * len(samples))
        )
        if len(samples) == 1:
            axes = axes[None, :]  # ensure 2D

        fig.suptitle(
            f"KL{kl} Cluster {cluster_id}  |  "
            f"pain={c_df['pain'].mean():.1f}  "
            f"jsn_lat={c_df['jsn_lat'].mean():.2f}  "
            f"jsn_med={c_df['jsn_med'].mean():.2f}  "
            f"n={len(c_df)}",
            fontsize=11
        )

        for row_idx, (_, sample) in enumerate(samples.iterrows()):
            img_path = Path(sample["image_path"]) if "image_path" in sample \
                       else _find_image(sample)

            if not img_path.exists():
                print(f"    Image not found: {img_path}")
                continue

            # Load and preprocess
            img_pil    = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img_pil).unsqueeze(0)

            # ── Attention Rollout ──────────────────────────────────────────
            rollout_map = rollout(img_tensor)
            rollout_overlay = overlay_heatmap(img_pil, rollout_map,
                                              colormap=cv2.COLORMAP_INFERNO)

            # ── GradCAM++ ──────────────────────────────────────────────────
            gradcam_map = gradcam(img_tensor)
            gradcam_overlay = overlay_heatmap(img_pil, gradcam_map,
                                              colormap=cv2.COLORMAP_JET)

            # ── Plot row ───────────────────────────────────────────────────
            # Original
            axes[row_idx, 0].imshow(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
            axes[row_idx, 0].set_title(
                f"Original\npain={sample['pain']:.0f} "
                f"jsn_m={sample['jsn_med']:.0f} "
                f"jsn_l={sample['jsn_lat']:.0f}",
                fontsize=8
            )
            axes[row_idx, 0].axis("off")

            # GradCAM++
            axes[row_idx, 1].imshow(gradcam_overlay)
            axes[row_idx, 1].set_title("GradCAM++", fontsize=8)
            axes[row_idx, 1].axis("off")

            # Attention Rollout
            axes[row_idx, 2].imshow(rollout_overlay)
            axes[row_idx, 2].set_title("Attention Rollout", fontsize=8)
            axes[row_idx, 2].axis("off")

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"KL{kl}_cluster{cluster_id}_xai.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved → {out_path.name}")


def _find_image(sample):
    """Reconstruct image path from cluster CSV row."""
    crop_dir = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/cropped_yolo_V00_images")
    knee_num  = "knee1" if sample["knee_side"] == "LEFT" else "knee0"
    pattern   = f"*{sample['subject_id']}*{knee_num}*.png"
    matches   = list(crop_dir.glob(pattern))
    return matches[0] if matches else Path("not_found")


def main():
    # Load cluster assignments
    cluster_df = pd.read_csv(CLUSTER_CSV)
    print(f"Loaded {len(cluster_df)} cluster assignments")

    # Reconstruct image paths if not in CSV
    if "image_path" not in cluster_df.columns:
        crop_dir = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/cropped_yolo_V00_images")
        def get_path(row):
            knee_num = "knee1" if row["knee_side"] == "LEFT" else "knee0"
            matches  = list(crop_dir.glob(f"*{row['subject_id']}*{knee_num}*.png"))
            return str(matches[0]) if matches else ""
        print("Reconstructing image paths...")
        cluster_df["image_path"] = cluster_df.apply(get_path, axis=1)

    # Load model
    model, preprocess = load_model()

    # Init XAI methods
    print("Initializing XAI methods...")
    rollout = AttentionRollout(model, discard_ratio=0.9)
    gradcam = GradCAMPlusPlus(model)

    # Run for each KL grade
    # Skip KL0 for now (55 clusters = too many to visualize)
    for kl in [1, 2, 3, 4]:
        visualize_kl_grade(kl, cluster_df, model, preprocess,
                           rollout, gradcam)

    # Clean up hooks
    rollout.remove_hooks()
    gradcam.remove_hooks()

    print(f"\nDone! Copy to local:")
    print(f"  scp -r e20378@ada:{OUTPUT_DIR} ./xai_results")


if __name__ == "__main__":
    main()