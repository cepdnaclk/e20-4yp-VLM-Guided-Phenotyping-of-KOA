"""
Preview first N DICOM files and their left/right crops.
Run this BEFORE dicom_to_png.py to verify the crop is correct.

Output: /scratch1/.../PNG/preview/
  - {filename}_full.png       <- full bilateral image
  - {filename}_LEFT.png       <- left knee crop
  - {filename}_RIGHT.png      <- right knee crop

Usage:
    python preview_crops.py
    python preview_crops.py --n 20   # preview more files
"""

import argparse
import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DICOM_DIR   = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/DICOM_V00")
PREVIEW_DIR = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/PNG/preview")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def dicom_to_array(dcm_path):
    dcm = pydicom.dcmread(str(dcm_path))
    arr = dcm.pixel_array.astype(np.float32)

    slope     = float(getattr(dcm, "RescaleSlope",     1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    arr = arr.astype(np.uint8)

    photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if "MONOCHROME1" in str(photometric):
        arr = 255 - arr

    return arr


def make_preview(arr, filename_stem):
    """
    Create a side-by-side preview image showing:
    full image | left crop | right crop
    with labels and a red midpoint line on the full image.
    """
    h, w = arr.shape[:2]
    mid = w // 2

    # Crops
    right_knee = arr[:, :mid]   # image left  = patient RIGHT
    left_knee  = arr[:, mid:]   # image right = patient LEFT

    # Convert all to RGB PIL images, resize to same height for display
    display_h = 400
    def to_pil(a):
        img = Image.fromarray(a).convert("RGB")
        ratio = display_h / img.height
        new_w = int(img.width * ratio)
        return img.resize((new_w, display_h), Image.LANCZOS)

    full_img  = to_pil(arr)
    left_img  = to_pil(left_knee)
    right_img = to_pil(right_knee)

    # Draw red midpoint line on full image
    draw = ImageDraw.Draw(full_img)
    mid_x = full_img.width // 2
    draw.line([(mid_x, 0), (mid_x, full_img.height)], fill=(255, 0, 0), width=2)

    # Label height
    label_h = 30
    total_w = full_img.width + left_img.width + right_img.width + 40
    total_h = display_h + label_h

    # Compose side-by-side canvas
    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))

    # Paste images
    x = 0
    canvas.paste(full_img,  (x, label_h)); x += full_img.width + 20
    canvas.paste(right_img, (x, label_h)); x += right_img.width + 20
    canvas.paste(left_img,  (x, label_h))

    # Draw labels
    draw = ImageDraw.Draw(canvas)
    x = 0
    labels = [
        (full_img.width,  "FULL (red line = split)"),
        (right_img.width, "RIGHT knee (image left half)"),
        (left_img.width,  "LEFT knee (image right half)"),
    ]
    for img_w, label in labels:
        draw.text((x + img_w//2, 10), label, fill=(255, 255, 100), anchor="mm")
        x += img_w + 20

    # Add filename at bottom
    info = f"{filename_stem}  |  original size: {w}x{h}px"
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((10, total_h - 14), info, fill=(180, 180, 180))

    return canvas


def main(n=10):
    dicom_files = sorted(DICOM_DIR.glob("*.dcm"))[:n]
    print(f"Previewing {len(dicom_files)} files → {PREVIEW_DIR}\n")

    for dcm_path in dicom_files:
        print(f"  Processing: {dcm_path.name}")
        try:
            arr     = dicom_to_array(dcm_path)
            preview = make_preview(arr, dcm_path.stem)

            out_path = PREVIEW_DIR / f"{dcm_path.stem}_preview.png"
            preview.save(str(out_path))
            print(f"    Saved: {out_path.name}  ({arr.shape[1]}x{arr.shape[0]}px)")

        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nDone! Open these files to check the crops look correct:")
    print(f"  {PREVIEW_DIR}")
    print(f"\nIf the split looks wrong (e.g. cuts through a knee),")
    print(f"the crop logic in dicom_to_png.py may need adjusting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of files to preview")
    args = parser.parse_args()
    main(args.n)