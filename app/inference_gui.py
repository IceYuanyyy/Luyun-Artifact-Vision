"""
inference_gui.py
-----------------
A simple Tkinter GUI to load a trained YOLO model and classify images.

Usage:
    python app/inference_gui.py

Features:
    - Load a .pt or .onnx model
    - Select an image to classify
    - Display Top-5 predictions
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np

# Attempt to import Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    messagebox.showerror("Error", "Please install ultralytics: pip install ultralytics")
    exit()

class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Luyun Artifact Vision - 文物识别系统")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.model = None
        self.image_path = None
        self.id_to_name = self._load_id_mapping()
        
        self._create_widgets()
    
    def _load_id_mapping(self):
        try:
            import json
            mapping_path = os.path.join("datasets", "id_to_name.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Mapping load error: {e}")
            return {}

    def _create_widgets(self):
        # ... (Same widget creation) ...
        # --- Top Frame: Model Loading ---
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="模型路径:").pack(side=tk.LEFT)
        self.model_entry = ttk.Entry(top_frame, width=50)
        self.model_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="选择模型", command=self._load_model).pack(side=tk.LEFT, padx=5)
        
        self.model_status = ttk.Label(top_frame, text="未加载模型", foreground="red")
        self.model_status.pack(side=tk.LEFT, padx=10)
        
        # --- Middle Frame: Image Display ---
        mid_frame = ttk.Frame(self.root, padding=10)
        mid_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Image
        self.image_label = ttk.Label(mid_frame, text="请选择图片", anchor="center", relief="sunken")
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Right: Results
        result_frame = ttk.Frame(mid_frame, width=300)
        result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        result_frame.pack_propagate(False)
        
        ttk.Label(result_frame, text="识别结果 (Top-5):", font=("Arial", 12, "bold")).pack(pady=10)
        self.result_text = tk.Text(result_frame, height=20, width=35, state=tk.DISABLED, font=("Consolas", 11))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # --- Bottom Frame: Actions ---
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X)
        
        ttk.Button(bottom_frame, text="选择图片", command=self._select_image).pack(side=tk.LEFT, padx=10)
        ttk.Button(bottom_frame, text="开始识别", command=self._run_inference).pack(side=tk.LEFT, padx=10)
        ttk.Button(bottom_frame, text="清空", command=self._clear).pack(side=tk.RIGHT, padx=10)

    # ... (Other methods) ...

    def _run_inference(self):
        if self.model is None:
            messagebox.showwarning("提示", "请先加载模型!")
            return
        if self.image_path is None:
            messagebox.showwarning("提示", "请先选择图片!")
            return
        
        try:
            results = self.model(self.image_path)
            
            # For classification, results[0].probs contains the probabilities
            probs = results[0].probs
            top5_indices = probs.top5
            top5_confs = probs.top5conf
            names = results[0].names
            
            # Display results
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=" * 30 + "\n")
            self.result_text.insert(tk.END, "       Top-5 分类结果\n")
            self.result_text.insert(tk.END, "=" * 30 + "\n\n")
            
            for i, (idx, conf) in enumerate(zip(top5_indices, top5_confs)):
                class_id = names[idx]
                conf_pct = float(conf) * 100
                
                # Retrieve Real Name from Mapping
                real_name = self.id_to_name.get(class_id, class_id)
                
                self.result_text.insert(tk.END, f"{i+1}. {real_name}\n")
                if real_name != class_id:
                     self.result_text.insert(tk.END, f"   (ID: {class_id})\n")
                self.result_text.insert(tk.END, f"   置信度: {conf_pct:.2f}%\n\n")
            
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
             messagebox.showerror("识别失败", f"推理出错: {e}")
    
    def _clear(self):
        self.image_label.config(image="", text="请选择图片")
        self.image_label.image = None
        self.image_path = None
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
