"""
export_model.py
----------------
Exports the trained YOLOv8 classification model to ONNX format.
"""

from ultralytics import YOLO
import os
import shutil

def export_model():
    # Paths
    # Note: Adjust these if your run name changes
    source_weights = "runs/classify/runs/classify/artifact_cls_run/weights/best.pt"
    dest_dir = "models"
    dest_name = "best.onnx"
    dest_path = os.path.join(dest_dir, dest_name)

    # Check if source exists
    if not os.path.exists(source_weights):
        print(f"❌ Error: Source weights not found at {source_weights}")
        return

    print(f"🔄 Loading model from {source_weights}...")
    model = YOLO(source_weights)

    print("📤 Starting export to ONNX...")
    # export() returns the path to the exported file
    exported_path = model.export(format="onnx")
    
    # Verify export
    if exported_path and os.path.exists(exported_path):
        print(f"✅ Export successful: {exported_path}")
        
        # Move to models directory
        os.makedirs(dest_dir, exist_ok=True)
        
        # The exported file is usually in the same dir as weights
        # We move it to our central 'models' dir
        try:
            shutil.copy(exported_path, dest_path)
            print(f"📂 Copied to: {dest_path}")
        except Exception as e:
            print(f"⚠️ Could not copy file: {e}")
    else:
        print("❌ Export failed.")

if __name__ == "__main__":
    export_model()
