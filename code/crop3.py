import cv2
import numpy as np
import pydicom
import os
from glob import glob

def get_knee_centers(img):
    # Smooth and threshold to find bone masses
    blurred = cv2.medianBlur(img, 11)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find the two legs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    if num_labels < 3: return None, None # Fallback if detection fails

    # Get two largest areas (excluding background)
    large_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2:] + 1
    det_centroids = centroids[large_indices]
    
    # Sort: Smallest X = Patient Right, Largest X = Patient Left
    sorted_centroids = det_centroids[np.argsort(det_centroids[:, 0])]
    return sorted_centroids[0], sorted_centroids[1]

def process_and_save(dicom_path, output_dir, target_size=750):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(float)
    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    
    base_name = os.path.basename(dicom_path).replace('.dcm', '')
    right_c, left_c = get_knee_centers(img)
    
    if right_c is None:
        print(f"Skipping {base_name}: Could not detect legs.")
        return

    for side, center in [('RIGHT', right_c), ('LEFT', left_c)]:
        cX, cY = int(center[0]), int(center[1])
        
        # Fixed Window Crop with Padding
        pad = target_size // 2
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        crop = padded[cY:cY+target_size, cX:cX+target_size]
        
        # Enhancement
        crop = cv2.medianBlur(crop, 3) # Remove grid lines
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(crop)
        final = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # Save with your naming idea
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_{side}.png"), final)

# --- Run on first 10 ---
in_dir = '/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/Visit01/knee_dicom_files'
out_dir = './vlm_ready_images'
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob(os.path.join(in_dir, "*.dcm")))[:10]
for f in files:
    process_and_save(f, out_dir)