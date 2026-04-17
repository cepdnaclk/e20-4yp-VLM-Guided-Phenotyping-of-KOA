import cv2
import numpy as np
import pydicom
import os
from glob import glob

def find_joint_center(half_img):
    h, w = half_img.shape
    
    # 1. Standard Pre-processing
    blurred = cv2.GaussianBlur(half_img, (7, 7), 0)
    
    # 2. Find the leg column (X-axis) using a simple threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(thresh)
    if M["m00"] == 0: return w//2, h//2, w//2
    cX = int(M["m10"] / M["m00"])
    
    # 3. Focus on a narrow strip where the bone is
    strip_w = 60
    x1, x2 = max(0, cX-strip_w//2), min(w, cX+strip_w//2)
    bone_column = half_img[:, x1:x2]
    
    # 4. SOBEL GRADIENT (Horizontal Edges)
    # This highlights the top of the tibia and bottom of the femur
    sobel_y = cv2.Sobel(bone_column, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = np.abs(sobel_y)
    
    # 5. Find the "Edge Peak" in the middle 40% of the image
    # The tibial plateau is a very sharp horizontal gradient change
    h_profile = np.mean(sobel_y, axis=1)
    search_start, search_end = int(h*0.3), int(h*0.7)
    
    # We find the strongest horizontal edge in the search area
    cY = np.argmax(h_profile[search_start:search_end]) + search_start
    
    # 6. Calculate bone width for zooming
    # We use the threshold mask at the joint height to see how wide the leg is
    row_at_joint = thresh[cY, :]
    bone_pixels = np.where(row_at_joint > 0)[0]
    cw = bone_pixels[-1] - bone_pixels[0] if len(bone_pixels) > 0 else w//2
    
    return cX, cY, cw

def test_first_10(input_folder, debug_folder):
    os.makedirs(debug_folder, exist_ok=True)
    
    all_files = sorted(glob(os.path.join(input_folder, "*.dcm")))[:10]
    
    for f_path in all_files:
        ds = pydicom.dcmread(f_path)
        img = ds.pixel_array.astype(float)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        
        base_name = os.path.basename(f_path).replace('.dcm', '')
        h, w = img.shape
        mid = w // 2

        for side in ['RIGHT', 'LEFT']:
            half = img[:, :mid] if side == 'RIGHT' else img[:, mid:]
            
            # Now we get cX, cY, and the bone width (cw)
            cX, cY, cw = find_joint_center(half)
            
            # --- DYNAMIC ZOOM LOGIC ---
            # We set the crop size relative to the bone width (1.8x the bone width)
            # This ensures that a thin leg and a thick leg look the same size!
            target_size = max(600, int(cw * 2.0)) 
            pad = target_size // 2
            
            # Create Debug Image
            debug_img = cv2.cvtColor(half, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_img, (cX, cY), 20, (0, 0, 255), -1) # Red Dot
            cv2.rectangle(debug_img, (cX-pad, cY-pad), (cX+pad, cY+pad), (0, 255, 0), 5) # Green Box
            
            # --- SAVE CROP ---
            padded = cv2.copyMakeBorder(half, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            # Adjusting indexing for the padded image
            cX_p, cY_p = cX + pad, cY + pad
            crop = padded[cY_p-pad:cY_p+pad, cX_p-pad:cX_p+pad]
            
            # Final Resize for VLM
            final = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            
            cv2.imwrite(os.path.join(debug_folder, f"{base_name}_{side}_debug.png"), debug_img)
            cv2.imwrite(os.path.join(debug_folder, f"{base_name}_{side}_crop.png"), final)

# Run it
input_dir = '/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset/Visit01/knee_dicom_files'
output_dir = '/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/debug_crops_test'
test_first_10(input_dir, output_dir)