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
            # 获取项目根目录（app 目录的父目录）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            mapping_path = os.path.join(project_root, "datasets", "id_to_name.json")
            
            print(f"[DEBUG] Loading mapping from: {mapping_path}")
            
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"[DEBUG] Loaded {len(data)} mappings")
                    return data
            else:
                print(f"[DEBUG] Mapping file not found at: {mapping_path}")
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


    def _browse_model(self):
        """打开文件对话框选择模型文件"""
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("YOLO 模型", "*.pt *.onnx"), ("所有文件", "*.*")]
        )
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)
            self._load_model_from_path(path)
    
    def _load_model(self):
        """从输入框路径加载模型，或弹出文件对话框"""
        path = self.model_entry.get().strip()
        if path and os.path.exists(path):
            self._load_model_from_path(path)
        else:
            self._browse_model()
    
    def _load_model_from_path(self, path):
        """实际加载模型"""
        try:
            self.model_status.config(text="加载中...", foreground="orange")
            self.root.update()
            
            # ONNX 模型需要显式指定任务类型
            if path.endswith('.onnx'):
                self.model = YOLO(path, task='classify')
            else:
                self.model = YOLO(path)
            
            self.model_status.config(text=f"已加载: {os.path.basename(path)}", foreground="green")
        except Exception as e:
            self.model = None
            self.model_status.config(text="加载失败", foreground="red")
            messagebox.showerror("加载失败", f"无法加载模型:\n{e}")
    
    def _select_image(self):
        """选择要识别的图片"""
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if path:
            self.image_path = path
            self._display_image(path)
    
    def _display_image(self, path):
        """在界面中显示图片"""
        try:
            img = Image.open(path)
            
            # 智能缩放以适应显示区域
            max_w, max_h = 500, 500
            img_ratio = img.width / img.height
            
            new_w = min(img.width, max_w)
            new_h = int(new_w / img_ratio)
            
            if new_h > max_h:
                new_h = max_h
                new_w = int(new_h * img_ratio)
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("图片加载失败", f"无法打开图片:\n{e}")

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
                
                # 去除可能的前缀 "__" 或 "_"，用于 ID 查找
                clean_id = class_id.lstrip('_')
                
                # 从映射中获取中文名称
                real_name = self.id_to_name.get(clean_id, None)
                
                if real_name:
                    self.result_text.insert(tk.END, f"{i+1}. {real_name}\n")
                    self.result_text.insert(tk.END, f"   (ID: {clean_id})\n")
                else:
                    self.result_text.insert(tk.END, f"{i+1}. {class_id}\n")
                
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
