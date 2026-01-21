
import os
import cv2
import shutil
import random
import albumentations as A
import numpy as np
from pathlib import Path

# --- Helper Functions for Non-ASCII Paths (Windows) ---
def cv2_imread(file_path):
    """Read image with non-ASCII path support."""
    try:
        # np.fromfile reads data, cv2.imdecode decodes it
        img_array = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def cv2_imwrite(file_path, img):
    """Write image with non-ASCII path support."""
    try:
        # cv2.imencode encodes to memory, tofile writes to disk
        is_success, img_array = cv2.imencode(os.path.splitext(file_path)[1], img)
        if is_success:
            img_array.tofile(file_path)
            return True
        return False
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

# --- Configuration ---
# Adapt paths to be absolute or relative to project root
SOURCE_DIR = os.path.join("datasets", "raw")
OUTPUT_DIR = os.path.join("datasets", "processed")

TARGET_COUNT = 50              # Target images per class
VAL_RATIO = 0.2                # 20% validation set

# --- Augmentation Pipeline ---
transform = A.Compose([
    A.Rotate(limit=30, p=0.7),                 # Random rotation
    A.HorizontalFlip(p=0.5),                   # Horizontal flip
    A.RandomBrightnessContrast(p=0.5),         # Brightness/Contrast
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Blur
    A.ISONoise(p=0.3),                         # ISO Noise
    A.Perspective(scale=(0.05, 0.1), p=0.3),   # Perspective transform
    A.Resize(224, 224)                         # Resize to YOLO cls default
])

def remove_watermark(img):
    """
    Simple watermark removal heuristic:
    Inpaint white/bright text in the bottom-right corner (common for Weibo/Baidu).
    """
    try:
        h, w = img.shape[:2]
        # Define ROI: Bottom-right 20%
        roi_x, roi_y = int(w * 0.8), int(h * 0.8)
        roi = img[roi_y:h, roi_x:w]
        
        if roi.size == 0: return img

        # 1. Convert ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 2. Threshold to find bright text (e.g., > 200)
        # Assuming watermark is whiteish
        _, mask = cv2.threshold(gray_roi, 215, 255, cv2.THRESH_BINARY)
        
        # 3. Dilate mask slightly to cover edges
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 4. Inpaint the ROI
        # INPAINT_TELEA or INPAINT_NS
        inpainted_roi = cv2.inpaint(roi, dilated_mask, 3, cv2.INPAINT_TELEA)
        
        # 5. Put back
        img[roi_y:h, roi_x:w] = inpainted_roi
        return img
    except Exception as e:
        print(f"Watermark removal error: {e}")
        return img

import json

# ... (Previous imports)

def process():
    # 1. Clean old data
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned existing directory: {OUTPUT_DIR}")
    
    # 2. Traverse categories
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        return

    categories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"🚀 Starting processing. Found {len(categories)} main categories.")

    id_to_name_map = {}  # Store ID -> Name mapping

    for cat in categories:
        cat_path = os.path.join(SOURCE_DIR, cat)
        # Traverse specific artifact folders (e.g., Era_Name_ShortID)
        artifacts = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
        
        for art in artifacts:
            # Extract ShortID as class name (e.g., 89f8c3)
            # Folder format assumption: Era_Name_ShortID
            parts = art.split('_')
            try:
                class_name = parts[-1] 
                # Extract real name (Middle part or full string minus ID)
                if len(parts) >= 3:
                     real_name = parts[1] # Era_Name_ID -> Name
                else:
                     real_name = art # Fallback
            except:
                class_name = art # Fallback
                real_name = art

            id_to_name_map[class_name] = real_name # Save mapping
            
            src_art_path = os.path.join(cat_path, art)
            
            # ... (Rest of processing) ...
            # Collect original images
            images = [f for f in os.listdir(src_art_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                continue

            # Create train/val directories
            train_dir = os.path.join(OUTPUT_DIR, 'train', class_name)
            val_dir = os.path.join(OUTPUT_DIR, 'val', class_name)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            print(f"Processing: {art} -> ID: {class_name} (Name: {real_name})")

            # --- Augment and Distribute ---
            generated_count = 0
            
            while generated_count < TARGET_COUNT:
                # ... (Same loop logic) ...
                if generated_count < len(images):
                    chosen_file = images[generated_count]
                    is_original = True
                else:
                    chosen_file = random.choice(images)
                    is_original = False

                img_path = os.path.join(src_art_path, chosen_file)
                # Use helper for reading
                img = cv2_imread(img_path)
                
                if img is None:
                    # ... (Error handling) ...
                    if chosen_file in images:
                        images.remove(chosen_file) 
                    if not images: break
                    continue

                # --- NEW: Watermark Removal ---
                if is_original:
                    img = remove_watermark(img)
                # ------------------------------

                # Determine split (Train vs Val)
                is_val = random.random() < VAL_RATIO
                target_folder = val_dir if is_val else train_dir
                
                if generated_count < len(images) and is_original: 
                    # First pass: Save Original (Resized)
                    try:
                        save_img = cv2.resize(img, (224, 224))
                        prefix = "orig"
                    except Exception as e:
                        print(f"Resize failed for {chosen_file}: {e}")
                        continue
                else:
                    # Augmentation
                    try:
                        save_img = transform(image=img)['image']
                        prefix = "aug"
                    except Exception as e:
                        print(f"Augmentation failed for {chosen_file}: {e}")
                        continue
                
                # Use helper for writing
                save_name = f"{prefix}_{generated_count}_{chosen_file}"
                save_path = os.path.join(target_folder, save_name)
                cv2_imwrite(save_path, save_img)
                generated_count += 1

    # Save Mapping
    mapping_path = os.path.join("datasets", "id_to_name.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_name_map, f, ensure_ascii=False, indent=2)
    print(f"✅ Mapping saved to {mapping_path}")
    print("✅ Data processing complete! Ready for training.")

if __name__ == "__main__":
    process()
