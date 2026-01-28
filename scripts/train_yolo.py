"""
train_yolo.py
--------------
Training launch script for YOLOv8 model (Classification).

Usage:
    python scripts/train_yolo.py

Description:
    This script initializes the YOLOv8-cls model and trains it using
    the dataset located in datasets/processed.
"""

from ultralytics import YOLO
import os

def train():
    # 1. Configuration
    DATASET_DIR = "datasets/processed"
    MODEL_NAME = "yolov8s-cls.pt"  # Small model for better fine-grained recognition
    EPOCHS = 100  # 细粒度分类需要更长训练时间
    IMG_SIZE = 224
    BATCH_SIZE = 16
    PROJECT_NAME = "Luyun-Artifact-Vision"
    RUN_NAME = "artifact_cls_run"

    # Absolute path to dataset for safety
    dataset_abs_path = os.path.abspath(DATASET_DIR)
    
    print(f"🚀 Starting YOLOv8 Classification Training...")
    print(f"Dataset: {dataset_abs_path}")
    print(f"Model: {MODEL_NAME}, Epochs: {EPOCHS}")

    # 2. Initialize Model
    # Load a pretrained YOLOv8n classification model
    model = YOLO(MODEL_NAME) 

    # 3. Train
    # Note: 'data' argument for classification expects the folder name containing 'train' and 'val'
    results = model.train(
        data=dataset_abs_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project="runs/classify",
        name=RUN_NAME,
        exist_ok=True,  # Overwrite updated run
        # device=0        # Let YOLO auto-select device
    )

    print("✅ Training Complete.")
    print(f"Best model saved to: runs/classify/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    train()
