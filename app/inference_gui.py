"""
inference_gui.py
-----------------
文物识别系统 - 现代化 UI 版本

特性：
    - 现代化深色主题界面
    - 模型下拉快速选择
    - 拖放图片支持
    - 批量识别
    - 快捷键支持
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import glob

# Attempt to import Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    messagebox.showerror("Error", "Please install ultralytics: pip install ultralytics")
    exit()

# 尝试导入拖放支持
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


# ==================== 现代化主题配置 ====================
class ModernTheme:
    # 主色调
    PRIMARY = "#6366F1"       # 靛蓝色 - 主色
    PRIMARY_DARK = "#4F46E5"  # 深靛蓝
    PRIMARY_LIGHT = "#818CF8" # 浅靛蓝
    
    # 背景色
    BG_DARK = "#0F172A"       # 深色背景
    BG_CARD = "#1E293B"       # 卡片背景
    BG_HOVER = "#334155"      # 悬停背景
    BG_INPUT = "#1E293B"      # 输入框背景
    
    # 文字颜色
    TEXT_PRIMARY = "#F8FAFC"   # 主要文字
    TEXT_SECONDARY = "#94A3B8" # 次要文字
    TEXT_MUTED = "#64748B"     # 弱化文字
    
    # 状态颜色
    SUCCESS = "#22C55E"       # 成功 - 绿色
    WARNING = "#F59E0B"       # 警告 - 橙色
    ERROR = "#EF4444"         # 错误 - 红色
    INFO = "#3B82F6"          # 信息 - 蓝色
    
    # 边框
    BORDER = "#334155"
    BORDER_FOCUS = "#6366F1"
    
    # 字体
    FONT_TITLE = ("Microsoft YaHei UI", 14, "bold")
    FONT_SUBTITLE = ("Microsoft YaHei UI", 11, "bold")
    FONT_BODY = ("Microsoft YaHei UI", 10)
    FONT_SMALL = ("Microsoft YaHei UI", 9)
    FONT_MONO = ("Consolas", 10)


class ScrollableFrame(ttk.Frame):
    """可滚动的Frame容器"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # 创建画布和滚动条
        self.canvas = tk.Canvas(self, bg=ModernTheme.BG_CARD, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        
        # 内部可滚动区域
        self.scrollable_frame = tk.Frame(self.canvas, bg=ModernTheme.BG_CARD)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # 不直接pack scrollable_frame，而是创建window
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # 布局画布和滚动条
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        
        self.v_scroll.pack(side="right", fill="y")
        self.h_scroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # 绑定鼠标滚轮
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 自动调整宽度
        self.canvas.bind('<Configure>', self._on_canvas_configure)

    def _on_canvas_configure(self, event):
        # 让内部frame宽度跟随canvas宽度，但设置最小宽度以触发水平滚动
        min_width = 450
        width = max(event.width, min_width)
        self.canvas.itemconfig(self.canvas_window, width=width)

    def _on_mousewheel(self, event):
        # 简单的鼠标滚轮滚动
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Luyun Artifact Vision - 文物智能识别系统")
        self.root.geometry("1250x750")
        self.root.minsize(900, 600)
        self.root.configure(bg=ModernTheme.BG_DARK)
        
        # 获取项目根目录
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # 状态变量
        self.model = None
        self.image_path = None
        self.image_list = []
        self.current_index = 0
        self.auto_recognize = tk.BooleanVar(value=True)
        self.id_to_name = self._load_id_mapping()
        self.available_models = self._scan_models()
        
        self._apply_theme()
        self._create_widgets()
        self._bind_shortcuts()
        self._setup_drag_drop()
        
        # 自动加载首选模型
        if self.available_models:
            self._auto_load_first_model()
    
    def _apply_theme(self):
        """应用现代化主题样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame 样式
        style.configure("TFrame", background=ModernTheme.BG_DARK)
        style.configure("Card.TFrame", background=ModernTheme.BG_CARD)
        
        # Label 样式
        style.configure("TLabel", 
                       background=ModernTheme.BG_DARK, 
                       foreground=ModernTheme.TEXT_PRIMARY,
                       font=ModernTheme.FONT_BODY)
        style.configure("Card.TLabel", background=ModernTheme.BG_CARD)
        style.configure("Title.TLabel", 
                       font=ModernTheme.FONT_TITLE,
                       foreground=ModernTheme.TEXT_PRIMARY)
        style.configure("Subtitle.TLabel", 
                       font=ModernTheme.FONT_SUBTITLE,
                       foreground=ModernTheme.TEXT_SECONDARY)
        style.configure("Status.TLabel", font=ModernTheme.FONT_SMALL)
        
        # Button 样式
        style.configure("TButton",
                       background=ModernTheme.PRIMARY,
                       foreground=ModernTheme.TEXT_PRIMARY,
                       font=ModernTheme.FONT_BODY,
                       padding=(12, 8),
                       borderwidth=0)
        style.map("TButton",
                 background=[('active', ModernTheme.PRIMARY_DARK), 
                           ('pressed', ModernTheme.PRIMARY_DARK)])
        
        # 次要按钮
        style.configure("Secondary.TButton",
                       background=ModernTheme.BG_HOVER,
                       foreground=ModernTheme.TEXT_PRIMARY)
        style.map("Secondary.TButton",
                 background=[('active', ModernTheme.BORDER)])
        
        # 小按钮
        style.configure("Small.TButton", padding=(8, 4), font=ModernTheme.FONT_SMALL)
        
        # Combobox 样式
        style.configure("TCombobox",
                       fieldbackground=ModernTheme.BG_INPUT,
                       background=ModernTheme.BG_HOVER,
                       foreground=ModernTheme.TEXT_PRIMARY,
                       arrowcolor=ModernTheme.TEXT_SECONDARY,
                       font=ModernTheme.FONT_BODY)
        
        # Checkbutton 样式
        style.configure("TCheckbutton",
                       background=ModernTheme.BG_DARK,
                       foreground=ModernTheme.TEXT_PRIMARY,
                       font=ModernTheme.FONT_SMALL)
        
        # LabelFrame 样式
        style.configure("TLabelframe",
                       background=ModernTheme.BG_CARD,
                       foreground=ModernTheme.TEXT_SECONDARY,
                       bordercolor=ModernTheme.BORDER)
        style.configure("TLabelframe.Label",
                       background=ModernTheme.BG_CARD,
                       foreground=ModernTheme.TEXT_SECONDARY,
                       font=ModernTheme.FONT_SUBTITLE)
        
        # Progressbar 样式
        style.configure("TProgressbar",
                       background=ModernTheme.PRIMARY,
                       troughcolor=ModernTheme.BG_HOVER)

    def _load_id_mapping(self):
        """加载ID到名称的映射"""
        try:
            import json
            mapping_path = os.path.join(self.project_root, "datasets", "id_to_name.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def _scan_models(self):
        """扫描项目中的模型文件"""
        models = {}
        search_patterns = [
            os.path.join(self.project_root, "*.pt"),
            os.path.join(self.project_root, "*.onnx"),
            os.path.join(self.project_root, "models", "*.pt"),
            os.path.join(self.project_root, "models", "*.onnx"),
            os.path.join(self.project_root, "runs", "**", "*.pt"),
            os.path.join(self.project_root, "runs", "**", "*.onnx"),
        ]
        
        for pattern in search_patterns:
            for path in glob.glob(pattern, recursive=True):
                name = os.path.basename(path)
                if name in models:
                    parent = os.path.basename(os.path.dirname(path))
                    name = f"{parent}/{name}"
                models[name] = path
        
        return models

    def _create_widgets(self):
        # ==================== 顶部标题栏 ====================
        header = tk.Frame(self.root, bg=ModernTheme.BG_CARD, height=60)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        # Logo 和标题
        title_frame = tk.Frame(header, bg=ModernTheme.BG_CARD)
        title_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        tk.Label(
            title_frame, 
            text="🏛️", 
            font=("Segoe UI Emoji", 24),
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.PRIMARY
        ).pack(side=tk.LEFT)
        
        tk.Label(
            title_frame,
            text="Luyun Artifact Vision",
            font=("Microsoft YaHei UI", 16, "bold"),
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_PRIMARY
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(
            title_frame,
            text="文物智能识别系统",
            font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_MUTED
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # 模型选择区域
        model_frame = tk.Frame(header, bg=ModernTheme.BG_CARD)
        model_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        tk.Label(
            model_frame,
            text="AI 模型",
            font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_MUTED
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self.model_var = tk.StringVar()
        model_names = list(self.available_models.keys())
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=model_names,
            state="readonly",
            width=25
        )
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        
        self.model_status = tk.Label(
            model_frame,
            text="●",
            font=("Segoe UI", 12),
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.ERROR
        )
        self.model_status.pack(side=tk.LEFT, padx=(8, 0))
        
        # ==================== 主内容区域 ====================
        main_container = tk.Frame(self.root, bg=ModernTheme.BG_DARK)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 左侧面板 - 图片列表
        left_panel = self._create_card(main_container, "📁 图片队列", width=220)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 图片列表框
        list_container = tk.Frame(left_panel, bg=ModernTheme.BG_CARD)
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.image_listbox = tk.Listbox(
            list_container,
            bg=ModernTheme.BG_DARK,
            fg=ModernTheme.TEXT_PRIMARY,
            selectbackground=ModernTheme.PRIMARY,
            selectforeground=ModernTheme.TEXT_PRIMARY,
            font=ModernTheme.FONT_MONO,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=ModernTheme.BORDER,
            highlightcolor=ModernTheme.PRIMARY,
            activestyle='none'
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_selected)
        
        scrollbar = ttk.Scrollbar(
            list_container,
            orient="vertical",
            command=self.image_listbox.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        
        # 图片操作按钮
        btn_frame = tk.Frame(left_panel, bg=ModernTheme.BG_CARD)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(btn_frame, text="+ 添加", style="Small.TButton", 
                  command=self._add_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="📂 文件夹", style="Small.TButton",
                  command=self._add_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="清空", style="Secondary.TButton",
                  command=self._clear_list).pack(side=tk.RIGHT)
        
        # 自动识别开关
        auto_frame = tk.Frame(left_panel, bg=ModernTheme.BG_CARD)
        auto_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Checkbutton(
            auto_frame,
            text="自动识别",
            variable=self.auto_recognize,
            style="TCheckbutton"
        ).pack(side=tk.LEFT)
        
        # 中间面板 - 图片预览
        center_panel = self._create_card(main_container, "🖼️ 图片预览")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 图片显示区域
        self.image_container = tk.Frame(center_panel, bg=ModernTheme.BG_DARK)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # 拖放提示
        self.drop_frame = tk.Frame(self.image_container, bg=ModernTheme.BG_DARK)
        self.drop_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            self.drop_frame,
            text="🖼️",
            font=("Segoe UI Emoji", 48),
            bg=ModernTheme.BG_DARK,
            fg=ModernTheme.TEXT_MUTED
        ).pack(pady=(80, 10))
        
        tk.Label(
            self.drop_frame,
            text="拖放图片到此处",
            font=ModernTheme.FONT_SUBTITLE,
            bg=ModernTheme.BG_DARK,
            fg=ModernTheme.TEXT_SECONDARY
        ).pack()
        
        tk.Label(
            self.drop_frame,
            text="或点击左侧添加按钮选择图片",
            font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_DARK,
            fg=ModernTheme.TEXT_MUTED
        ).pack(pady=(5, 0))
        
        # 实际的图片标签
        self.image_label = tk.Label(
            self.image_container,
            bg=ModernTheme.BG_DARK,
            anchor="center"
        )
        
        # 导航栏
        nav_frame = tk.Frame(center_panel, bg=ModernTheme.BG_CARD)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(nav_frame, text="◀", width=4, style="Secondary.TButton",
                  command=self._prev_image).pack(side=tk.LEFT, padx=(0, 10))
        
        self.nav_label = tk.Label(
            nav_frame,
            text="0 / 0",
            font=ModernTheme.FONT_BODY,
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_SECONDARY
        )
        self.nav_label.pack(side=tk.LEFT, expand=True)
        
        ttk.Button(nav_frame, text="▶", width=4, style="Secondary.TButton",
                  command=self._next_image).pack(side=tk.RIGHT, padx=(10, 0))
        
        # 操作按钮
        action_frame = tk.Frame(center_panel, bg=ModernTheme.BG_CARD)
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(action_frame, text="🔍 开始识别", 
                  command=self._run_inference).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="📊 批量识别", style="Secondary.TButton",
                  command=self._batch_inference).pack(side=tk.LEFT)
        
        # 右侧面板 - 识别结果
        right_panel = self._create_card(main_container, "📋 识别结果", width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 结果显示 - 使用 ScrollableFrame
        self.result_scroll_frame = ScrollableFrame(right_panel)
        self.result_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))
        
        # 实际的内容容器是 scrollable_frame
        self.result_container = self.result_scroll_frame.scrollable_frame
        
        # 占位提示
        self.result_placeholder = tk.Label(
            self.result_container,
            text="选择图片后\n将在此显示识别结果",
            font=ModernTheme.FONT_BODY,
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_MUTED,
            justify="center"
        )
        self.result_placeholder.pack(pady=50)
        
        # 结果列表容器
        self.result_list_frame = tk.Frame(self.result_container, bg=ModernTheme.BG_CARD)
        
        # ==================== 底部状态栏 ====================
        footer = tk.Frame(self.root, bg=ModernTheme.BG_CARD, height=35)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        self.status_label = tk.Label(
            footer,
            text="就绪 | 快捷键: Ctrl+O 添加图片, Ctrl+R 识别, ←→ 切换图片",
            font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_MUTED,
            anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=8)
        
        self.model_info_label = tk.Label(
            footer,
            text="",
            font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_MUTED,
            anchor="e"
        )
        self.model_info_label.pack(side=tk.RIGHT, padx=15, pady=8)

    def _create_card(self, parent, title, width=None):
        """创建卡片式容器"""
        card = tk.Frame(parent, bg=ModernTheme.BG_CARD)
        if width:
            card.config(width=width)
            card.pack_propagate(False)
        
        # 卡片标题
        title_frame = tk.Frame(card, bg=ModernTheme.BG_CARD)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            title_frame,
            text=title,
            font=ModernTheme.FONT_SUBTITLE,
            bg=ModernTheme.BG_CARD,
            fg=ModernTheme.TEXT_SECONDARY
        ).pack(side=tk.LEFT)
        
        return card

    def _bind_shortcuts(self):
        """绑定快捷键"""
        self.root.bind("<Control-o>", lambda e: self._add_images())
        self.root.bind("<Control-O>", lambda e: self._add_images())
        self.root.bind("<Control-r>", lambda e: self._run_inference())
        self.root.bind("<Control-R>", lambda e: self._run_inference())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())

    def _setup_drag_drop(self):
        """设置拖放支持"""
        if DND_AVAILABLE and hasattr(self.root, 'drop_target_register'):
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_drop)

    def _on_drop(self, event):
        """处理拖放事件"""
        files = self.root.tk.splitlist(event.data)
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
        
        added = 0
        for f in files:
            if os.path.isfile(f) and f.lower().endswith(image_extensions):
                if f not in self.image_list:
                    self.image_list.append(f)
                    self.image_listbox.insert(tk.END, os.path.basename(f))
                    added += 1
            elif os.path.isdir(f):
                for ext in image_extensions:
                    for img in glob.glob(os.path.join(f, f"*{ext}")):
                        if img not in self.image_list:
                            self.image_list.append(img)
                            self.image_listbox.insert(tk.END, os.path.basename(img))
                            added += 1
        
        if added > 0:
            self._update_nav_label()
            self._update_status(f"已添加 {added} 张图片")
            if self.image_list and self.current_index == 0:
                self.image_listbox.selection_set(0)
                self._display_current_image()

    def _auto_load_first_model(self):
        """自动加载首选模型"""
        preferred = ['best.onnx', 'best.pt', 'yolov8s-cls.pt']
        selected = None
        
        for pref in preferred:
            for name in self.available_models:
                if pref in name:
                    selected = name
                    break
            if selected:
                break
        
        if not selected and self.available_models:
            selected = list(self.available_models.keys())[0]
        
        if selected:
            self.model_var.set(selected)
            self._on_model_selected(None)

    def _on_model_selected(self, event):
        """模型选择事件"""
        name = self.model_var.get()
        if name and name in self.available_models:
            path = self.available_models[name]
            self._load_model_from_path(path)

    def _load_model_from_path(self, path):
        """异步加载模型"""
        self.model_status.config(fg=ModernTheme.WARNING)
        self.model_info_label.config(text=f"正在加载模型: {os.path.basename(path)}...")
        self._update_status("正在加载模型，请稍候...")
        self.model_combo.config(state="disabled")
        self.root.update()
        
        # 启动后台线程加载模型
        threading.Thread(target=self._load_model_task, args=(path,), daemon=True).start()

    def _load_model_task(self, path):
        """后台加载模型任务"""
        try:
            if path.endswith('.onnx'):
                model = YOLO(path, task='classify')
            else:
                model = YOLO(path)
            # 加载完成，在主线程更新UI
            self.root.after(0, self._on_model_loaded, model, path, None)
        except Exception as e:
            # 加载失败，在主线程显示错误
            self.root.after(0, self._on_model_loaded, None, path, str(e))

    def _on_model_loaded(self, model, path, error):
        """模型加载回调"""
        self.model_combo.config(state="readonly")
        
        if model:
            self.model = model
            self.model_status.config(fg=ModernTheme.SUCCESS)
            self.model_info_label.config(text=f"模型: {os.path.basename(path)}")
            self._update_status("模型加载成功")
            
            # 如果开启了自动识别且当前有图片，尝试识别
            if self.auto_recognize.get() and self.image_list and self.current_index >= 0:
                self._run_inference()
        else:
            self.model = None
            self.model_status.config(fg=ModernTheme.ERROR)
            self.model_info_label.config(text="模型加载失败")
            self._update_status("模型加载失败")
            messagebox.showerror("加载失败", f"无法加载模型:\n{error}")

    def _add_images(self):
        """添加图片"""
        paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"),
                ("所有文件", "*.*")
            ]
        )
        
        added = 0
        for path in paths:
            if path not in self.image_list:
                self.image_list.append(path)
                self.image_listbox.insert(tk.END, os.path.basename(path))
                added += 1
        
        if added > 0:
            self._update_nav_label()
            self._update_status(f"已添加 {added} 张图片")
            if len(self.image_list) == added:
                self.image_listbox.selection_set(0)
                self._display_current_image()
                if self.auto_recognize.get() and self.model:
                    self._run_inference()

    def _add_folder(self):
        """添加文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.gif')
        added = 0
        
        for ext in exts:
            for path in glob.glob(os.path.join(folder, ext)):
                if path not in self.image_list:
                    self.image_list.append(path)
                    self.image_listbox.insert(tk.END, os.path.basename(path))
                    added += 1
                    # 每添加10张图片更新一次UI，防止完全卡死
                    if added % 10 == 0:
                        self.root.update()
        
        if added > 0:
            self._update_nav_label()
            self._update_status(f"从文件夹添加 {added} 张图片")
            self.image_listbox.selection_set(0)
            self._display_current_image()

    def _clear_list(self):
        """清空列表"""
        self.image_list.clear()
        self.image_listbox.delete(0, tk.END)
        self.current_index = 0
        self._update_nav_label()
        self._clear_display()

    def _on_image_selected(self, event):
        """图片选择事件"""
        selection = self.image_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self._display_current_image()
            if self.auto_recognize.get() and self.model:
                self._run_inference()

    def _display_current_image(self):
        """显示当前图片"""
        if 0 <= self.current_index < len(self.image_list):
            path = self.image_list[self.current_index]
            self.image_path = path
            self._display_image(path)
            self._update_nav_label()

    def _display_image(self, path):
        """显示图片"""
        try:
            img = Image.open(path)
            
            # 智能缩放
            max_w, max_h = 500, 400
            img_ratio = img.width / img.height
            
            new_w = min(img.width, max_w)
            new_h = int(new_w / img_ratio)
            
            if new_h > max_h:
                new_h = max_h
                new_w = int(new_h * img_ratio)
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # 切换显示
            self.drop_frame.pack_forget()
            self.image_label.pack(fill=tk.BOTH, expand=True)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("图片加载失败", f"无法打开图片:\n{e}")

    def _clear_display(self):
        """清空显示"""
        self.image_label.config(image="")
        self.image_label.image = None
        self.image_label.pack_forget()
        self.drop_frame.pack(fill=tk.BOTH, expand=True)
        self.image_path = None
        
        # 清空结果
        for widget in self.result_list_frame.winfo_children():
            widget.destroy()
        self.result_list_frame.pack_forget()
        self.result_placeholder.pack(pady=50)

    def _update_nav_label(self):
        """更新导航"""
        total = len(self.image_list)
        current = self.current_index + 1 if total > 0 else 0
        self.nav_label.config(text=f"{current} / {total}")

    def _update_status(self, text):
        """更新状态"""
        self.status_label.config(text=text)

    def _prev_image(self):
        """上一张"""
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)
            self._display_current_image()
            if self.auto_recognize.get() and self.model:
                self._run_inference()

    def _next_image(self):
        """下一张"""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)
            self._display_current_image()
            if self.auto_recognize.get() and self.model:
                self._run_inference()

    def _run_inference(self):
        """异步运行推理"""
        if self.model is None:
            messagebox.showwarning("提示", "请先选择模型!")
            return
        if self.image_path is None:
            messagebox.showwarning("提示", "请先选择图片!")
            return
        
        # 防止重复点击
        if getattr(self, '_is_inferencing', False):
            return
            
        self._is_inferencing = True
        self._update_status("正在识别...")
        
        # 启动后台线程进行推理
        threading.Thread(target=self._run_inference_task, args=(self.image_path,), daemon=True).start()

    def _run_inference_task(self, image_path):
        """后台推理任务"""
        try:
            results = self.model(image_path)
            # 在主线程处理结果
            self.root.after(0, self._on_inference_complete, results, None)
        except Exception as e:
            # 在主线程显示错误
            self.root.after(0, self._on_inference_complete, None, str(e))

    def _on_inference_complete(self, results, error):
        """推理完成回调"""
        self._is_inferencing = False
        
        if error:
            self._update_status("识别失败")
            messagebox.showerror("识别失败", f"推理出错: {error}")
            return
            
        try:
            probs = results[0].probs
            top5_indices = probs.top5
            top5_confs = probs.top5conf
            names = results[0].names
            
            # 清空旧结果
            for widget in self.result_list_frame.winfo_children():
                widget.destroy()
            
            self.result_placeholder.pack_forget()
            self.result_list_frame.pack(fill=tk.BOTH, expand=True)
            
            # 显示结果
            rank_colors = [ModernTheme.PRIMARY, "#8B5CF6", "#EC4899", ModernTheme.TEXT_SECONDARY, ModernTheme.TEXT_MUTED]
            
            for i, (idx, conf) in enumerate(zip(top5_indices, top5_confs)):
                class_id = names[idx]
                conf_pct = float(conf) * 100
                clean_id = class_id.lstrip('_')
                real_name = self.id_to_name.get(clean_id, class_id)
                
                # 单条结果卡片
                item = tk.Frame(self.result_list_frame, bg=ModernTheme.BG_DARK, pady=8, padx=8)
                item.pack(fill=tk.X, pady=3)
                
                # 排名
                rank = tk.Label(
                    item,
                    text=f"#{i+1}",
                    font=("Microsoft YaHei UI", 11, "bold"),
                    bg=ModernTheme.BG_DARK,
                    fg=rank_colors[i],
                    width=3
                )
                rank.pack(side=tk.LEFT)
                
                # 信息区
                info = tk.Frame(item, bg=ModernTheme.BG_DARK)
                info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
                
                tk.Label(
                    info,
                    text=real_name,
                    font=ModernTheme.FONT_BODY,
                    bg=ModernTheme.BG_DARK,
                    fg=ModernTheme.TEXT_PRIMARY,
                    anchor="w",
                    wraplength=400,  # 允许更长的换行
                    justify="left"
                ).pack(fill=tk.X)
                
                if real_name != class_id:
                    tk.Label(
                        info,
                        text=f"ID: {clean_id}",
                        font=ModernTheme.FONT_SMALL,
                        bg=ModernTheme.BG_DARK,
                        fg=ModernTheme.TEXT_MUTED,
                        anchor="w"
                    ).pack(fill=tk.X)
                
                # 置信度
                conf_color = ModernTheme.SUCCESS if conf_pct >= 70 else (ModernTheme.WARNING if conf_pct >= 40 else ModernTheme.TEXT_MUTED)
                tk.Label(
                    item,
                    text=f"{conf_pct:.1f}%",
                    font=("Microsoft YaHei UI", 11, "bold"),
                    bg=ModernTheme.BG_DARK,
                    fg=conf_color
                ).pack(side=tk.RIGHT)
            
            self._update_status("识别完成")
            
        except Exception as e:
            self._update_status("处理结果出错")
            messagebox.showerror("错误", f"处理结果失败: {e}")

    def _batch_inference(self):
        """批量识别"""
        if self.model is None:
            messagebox.showwarning("提示", "请先选择模型!")
            return
        if not self.image_list:
            messagebox.showwarning("提示", "请先添加图片!")
            return
        
        # 创建结果窗口
        win = tk.Toplevel(self.root)
        win.title("批量识别结果")
        win.geometry("750x550")
        win.configure(bg=ModernTheme.BG_DARK)
        
        # 标题
        tk.Label(
            win,
            text="📊 批量识别结果",
            font=ModernTheme.FONT_TITLE,
            bg=ModernTheme.BG_DARK,
            fg=ModernTheme.TEXT_PRIMARY
        ).pack(pady=15)
        
        # 表格
        columns = ("序号", "图片名称", "识别结果", "置信度")
        tree = ttk.Treeview(win, columns=columns, show="headings", height=18)
        
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column("序号", width=60, anchor="center")
        tree.column("图片名称", width=220)
        tree.column("识别结果", width=320)
        tree.column("置信度", width=100, anchor="center")
        
        tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
        
        # 进度条
        progress = ttk.Progressbar(win, mode='determinate', maximum=len(self.image_list))
        progress.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # 批量处理
        for i, img_path in enumerate(self.image_list):
            try:
                results = self.model(img_path)
                probs = results[0].probs
                top1_idx = probs.top1
                top1_conf = float(probs.top1conf) * 100
                names = results[0].names
                
                class_id = names[top1_idx]
                clean_id = class_id.lstrip('_')
                real_name = self.id_to_name.get(clean_id, class_id)
                
                tree.insert("", tk.END, values=(
                    i + 1,
                    os.path.basename(img_path),
                    real_name,
                    f"{top1_conf:.1f}%"
                ))
            except Exception as e:
                tree.insert("", tk.END, values=(
                    i + 1,
                    os.path.basename(img_path),
                    f"错误: {str(e)[:30]}",
                    "-"
                ))
            
            progress['value'] = i + 1
            win.update()
        
        self._update_status(f"批量识别完成: {len(self.image_list)} 张图片")


def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    app = InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
