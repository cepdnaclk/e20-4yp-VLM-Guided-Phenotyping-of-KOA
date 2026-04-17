import cv2
import numpy as np
import pydicom
import os

def dynamic_localize(img, side):
    """
    Finds the center of the knee joint by analyzing bone density profiles.
    """
    h, w = img.shape
    mid_w = w // 2
    
    # 1. Initial Side Split
    half = img[:, :mid_w] if side.upper() == 'RIGHT' else img[:, mid_w:]
    
    # 2. Find Horizontal Center (X)
    # Sum intensities vertically to find the 'peak' density of the leg
    v_profile = np.sum(half[int(h*0.2):int(h*0.8), :], axis=0)
    cX = np.argmax(v_profile)
    
    # 3. Find Vertical Center (Y) 
    # The joint space is usually in the middle 40-60% of the image height
    # We look for the peak density in the horizontal profile within the leg column
    h_profile = np.sum(half[:, max(0, cX-50):min(half.shape[1], cX+50)], axis=1)
    cY = np.argmax(h_profile[int(h*0.3):int(h*0.7)]) + int(h*0.3)
    
    return half, cX, cY

def preprocess_knee_xray(dicom_path, side, output_size=(224, 224), zoom_factor=0.6):
    # 1. Load DICOM
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(float)
    
    # 2. Global Normalization (0-255)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype(np.uint8)
    
    # 3. Dynamic Centering Logic
    half_img, cX, cY = dynamic_localize(img, side)
    
    # 4. Zoom/Crop Logic
    # We take a square crop based on the smaller dimension (usually width) 
    # multiplied by a zoom factor to focus strictly on the joint.
    h, w = half_img.shape
    crop_dim = int(min(h, w) * zoom_factor)
    
    y1 = max(0, cY - crop_dim // 2)
    y2 = min(h, y1 + crop_dim)
    x1 = max(0, cX - crop_dim // 2)
    x2 = min(w, x1 + crop_dim)
    
    crop = half_img[y1:y2, x1:x2]
    
    # 5. CLAHE Enhancement (Applied to the crop for local contrast)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(crop)
    
    # 6. Resize to VLM Input Size (224, 224)
    resized_img = cv2.resize(enhanced_img, output_size, interpolation=cv2.INTER_LANCZOS4)
    
    return resized_img

# --- Execution ---
input_path = 'FYP/Dataset/Visit01/knee_dicom_files/00m_0.C.2_9001400_20041202_file.dcm'
processed_knee = preprocess_knee_xray(input_path, 'RIGHT')
cv2.imwrite('right_knee_centered_zoom1.png', processed_knee)