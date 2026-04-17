import cv2
import numpy as np
import pydicom
import os
from glob import glob

def dynamic_localize(img, side):
    h, w = img.shape
    mid_w = w // 2
    half = img[:, :mid_w] if side.upper() == 'RIGHT' else img[:, mid_w:]
    
    # Analyze density to find the center of the bone column
    v_profile = np.sum(half[int(h*0.2):int(h*0.8), :], axis=0)
    cX = np.argmax(v_profile)
    
    # Find the joint space (peak density in the middle vertical section)
    h_profile = np.sum(half[:, max(0, cX-50):min(half.shape[1], cX+50)], axis=1)
    cY = np.argmax(h_profile[int(h*0.3):int(h*0.7)]) + int(h*0.3)
    
    return half, cX, cY

def preprocess_knee_xray(dicom_path, side, output_size=(224, 224), zoom_factor=0.6):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(float)
    
    # 1. Normalization
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype(np.uint8)
    
    # 2. De-striping: Apply Median Filter to remove vertical grid lines
    # A 3x3 or 5x5 kernel is usually enough to kill the lines without blurring the bone
    img = cv2.medianBlur(img, 3)
    
    # 3. Dynamic Centering
    half_img, cX, cY = dynamic_localize(img, side)
    
    # 4. Zoom & Crop
    h, w = half_img.shape
    crop_dim = int(min(h, w) * zoom_factor)
    y1, y2 = max(0, cY - crop_dim // 2), min(h, cY + crop_dim // 2)
    x1, x2 = max(0, cX - crop_dim // 2), min(w, cX + crop_dim // 2)
    crop = half_img[y1:y2, x1:x2]
    
    # 5. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(crop)
    
    return cv2.resize(enhanced_img, output_size, interpolation=cv2.INTER_LANCZOS4), (cX, cY)

# --- Batch Run ---
input_dir = '/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/Visit01/knee_dicom_files'
output_dir = '/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/preprocessed_test'
os.makedirs(output_dir, exist_ok=True)

# Get first 10 DICOM files
dicom_files = sorted(glob(os.path.join(input_dir, "*.dcm")))[:10]

print(f"Processing {len(dicom_files)} images...")

for f_path in dicom_files:
    fname = os.path.basename(f_path)
    # Testing both sides for each image since they are bilateral
    for side in ['LEFT', 'RIGHT']:
        processed, center = preprocess_knee_xray(f_path, side)
        
        # Save the result
        out_name = f"{fname.replace('.dcm', '')}_{side}.png"
        cv2.imwrite(os.path.join(output_dir, out_name), processed)
        
print(f"Done! Check the results in: {output_dir}")