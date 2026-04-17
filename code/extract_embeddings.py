"""
BiomedCLIP Embedding Extractor for Knee OAI Dataset
=====================================================
1. Matches each cropped knee image to its metadata row in master_final_V00.csv
2. Extracts 512-dim visual embeddings using BiomedCLIP
3. Saves embeddings + metadata to a single .npz file for clustering

Usage:
    pip install open_clip_torch torch torchvision pandas tqdm pillow
    python extract_embeddings.py

Output:
    /scratch1/.../embeddings/
        embeddings_V00.npz   <- contains:
            embeddings   : (N, 512) float32
            image_paths  : (N,)     str
            subject_ids  : (N,)     str
            knee_sides   : (N,)     str  LEFT/RIGHT
            kl_grades    : (N,)     int
            pain_scores  : (N,)     float
            bmis         : (N,)     float
            ages         : (N,)     int
            jsn_lateral  : (N,)     float
            jsn_medial   : (N,)     float
            matched      : (N,)     bool  True if metadata found
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import open_clip

# ── Paths ─────────────────────────────────────────────────────────────────────
CROP_DIR    = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/cropped_yolo_V00_images")
METADATA    = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/master_final_V00.csv")
OUTPUT_DIR  = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 16
#DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE      = "cpu"
# knee0 = RIGHT, knee1 = LEFT  (YOLO detects left-to-right in image)
KNEE_MAP = {"knee0": "RIGHT", "knee1": "LEFT", "knee2": None}

# ── Load metadata ─────────────────────────────────────────────────────────────
def load_metadata(csv_path):
    df = pd.read_csv(csv_path)

    # Parse INTERVIEW_DATE to YYYYMMDD string for matching
    # Original format: 03-NOV-05  →  20051103
    df["DATE_STR"] = pd.to_datetime(
        df["INTERVIEW_DATE"], format="%d-%b-%y"
    ).dt.strftime("%Y%m%d")

    df["SRC_SUBJECT_ID"] = df["SRC_SUBJECT_ID"].astype(str)
    df["KL_GRADE"]       = pd.to_numeric(df["KL_GRADE"],       errors="coerce")
    df["MATCHED_WOMAC_PAIN"] = pd.to_numeric(df["MATCHED_WOMAC_PAIN"], errors="coerce")
    df["BMI"]            = pd.to_numeric(df["BMI"],            errors="coerce")
    df["AGEYEARS"]       = pd.to_numeric(df["AGEYEARS"],       errors="coerce")
    df["JSN_LATERAL"]    = pd.to_numeric(df["JSN_LATERAL"],    errors="coerce")
    df["JSN_MEDIAL"]     = pd.to_numeric(df["JSN_MEDIAL"],     errors="coerce")

    # Build lookup: (subject_id, date_str, knee_side) → row
    df["_key"] = (
        df["SRC_SUBJECT_ID"] + "_" +
        df["DATE_STR"]       + "_" +
        df["KNEE_SIDE"]
    )
    lookup = df.set_index("_key").to_dict("index")
    print(f"Metadata loaded: {len(df)} rows, {df['_key'].nunique()} unique keys")
    return lookup


def parse_filename(png_path):
    """
    00m_0.C.2_9000296_20040729_file_knee0_244x244.png
    → subject_id='9000296', date='20040729', knee_side='LEFT'
    """
    stem  = png_path.stem  # remove .png
    parts = stem.split("_")
    # parts: ['00m', '0.C.2', '9000296', '20040729', 'file', 'knee0', '244x244']
    subject_id = parts[2]
    date_str   = parts[3]
    knee_key   = parts[5]           # 'knee0' or 'knee1'
    knee_side  = KNEE_MAP[knee_key] # 'LEFT'  or 'RIGHT'
    return subject_id, date_str, knee_side


# ── Load BiomedCLIP ───────────────────────────────────────────────────────────
def load_model():
    print(f"Loading BiomedCLIP on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.eval().to(DEVICE)
    print("BiomedCLIP loaded.")
    return model, preprocess


# ── Extract embeddings ────────────────────────────────────────────────────────
def extract_embeddings(model, preprocess, image_paths):
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting embeddings"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception as e:
                print(f"  Failed to load {p.name}: {e}")
                imgs.append(torch.zeros(3, 224, 224))  # zero placeholder

        batch = torch.stack(imgs).to(DEVICE)

        with torch.no_grad(), torch.cuda.amp.autocast():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalise

        all_embeddings.append(features.cpu().float().numpy())

    return np.vstack(all_embeddings)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load metadata lookup
    lookup = load_metadata(METADATA)

    # Collect all PNG files
    png_files = sorted(CROP_DIR.glob("*.png"))
    print(f"\nFound {len(png_files)} cropped knee images")

    # Match each image to metadata
    records = []
    unmatched = []

    for p in png_files:
        try:
            subject_id, date_str, knee_side = parse_filename(p)
            key = f"{subject_id}_{date_str}_{knee_side}"
            row = lookup.get(key, None)

            if row:
                records.append({
                    "path":       p,
                    "subject_id": subject_id,
                    "knee_side":  knee_side,
                    "kl_grade":   row.get("KL_GRADE",           np.nan),
                    "pain":       row.get("MATCHED_WOMAC_PAIN",  np.nan),
                    "bmi":        row.get("BMI",                 np.nan),
                    "age":        row.get("AGEYEARS",            np.nan),
                    "jsn_lat":    row.get("JSN_LATERAL",         np.nan),
                    "jsn_med":    row.get("JSN_MEDIAL",          np.nan),
                    "matched":    True,
                })
            else:
                unmatched.append(p.name)
                records.append({
                    "path":       p,
                    "subject_id": subject_id,
                    "knee_side":  knee_side,
                    "kl_grade":   np.nan,
                    "pain":       np.nan,
                    "bmi":        np.nan,
                    "age":        np.nan,
                    "jsn_lat":    np.nan,
                    "jsn_med":    np.nan,
                    "matched":    False,
                })
        except Exception as e:
            print(f"  Parse error {p.name}: {e}")

    matched_count = sum(1 for r in records if r["matched"])
    print(f"Matched:   {matched_count} / {len(records)}")
    print(f"Unmatched: {len(unmatched)}")
    if unmatched[:5]:
        print(f"  e.g. {unmatched[:5]}")

    # Load model
    model, preprocess = load_model()

    # Extract embeddings
    image_paths = [r["path"] for r in records]
    embeddings  = extract_embeddings(model, preprocess, image_paths)

    # Save everything to .npz
    out_path = OUTPUT_DIR / "embeddings_V00.npz"
    np.savez(
        out_path,
        embeddings   = embeddings,
        image_paths  = np.array([str(r["path"])   for r in records]),
        subject_ids  = np.array([r["subject_id"]  for r in records]),
        knee_sides   = np.array([r["knee_side"]   for r in records]),
        kl_grades    = np.array([r["kl_grade"]    for r in records], dtype=float),
        pain_scores  = np.array([r["pain"]        for r in records], dtype=float),
        bmis         = np.array([r["bmi"]         for r in records], dtype=float),
        ages         = np.array([r["age"]         for r in records], dtype=float),
        jsn_lateral  = np.array([r["jsn_lat"]     for r in records], dtype=float),
        jsn_medial   = np.array([r["jsn_med"]     for r in records], dtype=float),
        matched      = np.array([r["matched"]     for r in records], dtype=bool),
    )

    print(f"\nSaved embeddings → {out_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"\nVerify with:")
    print(f"  python -c \"import numpy as np; d=np.load('{out_path}', allow_pickle=True); print(d['embeddings'].shape)\"")


if __name__ == "__main__":
    main()