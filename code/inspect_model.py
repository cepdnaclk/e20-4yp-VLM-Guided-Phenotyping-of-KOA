"""
Inspect BiomedCLIP architecture to get exact layer names for freezing.
Run this first before fine-tuning.

Usage:
    python inspect_model.py
"""

import open_clip
import torch

model, _, _ = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

visual = model.visual
print("=" * 60)
print("VISUAL ENCODER ARCHITECTURE")
print("=" * 60)

# Top level
for name, module in visual.named_children():
    print(f"\n[{name}]  {type(module).__name__}")

print("\n" + "=" * 60)
print("TRANSFORMER BLOCKS (visual.trunk.blocks)")
print("=" * 60)

trunk = visual.trunk
for name, module in trunk.named_children():
    print(f"  [{name}]  {type(module).__name__}")
    if name == "blocks":
        for i, block in enumerate(module):
            param_count = sum(p.numel() for p in block.parameters())
            print(f"    block[{i}]  params={param_count:,}")

print("\n" + "=" * 60)
print("ALL NAMED PARAMETERS (first 40)")
print("=" * 60)
params = list(model.visual.named_parameters())
for name, param in params[:40]:
    print(f"  {name:60s}  {str(list(param.shape)):20s}  requires_grad={param.requires_grad}")

print(f"\n  ... and {len(params)-40} more parameters")

print("\n" + "=" * 60)
print("TOTAL PARAMETER COUNT")
print("=" * 60)
total     = sum(p.numel() for p in model.visual.parameters())
trainable = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)
print(f"  Total     : {total:,}")
print(f"  Trainable : {trainable:,}")