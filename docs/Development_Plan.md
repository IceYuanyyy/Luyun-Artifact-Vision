# Luyun-Artifact-Vision 开发计划

## 1. 数据流架构

```mermaid
graph LR
    subgraph "第一阶段: 数据准备 (Data Preparation)"
        Raw[原始数据 output/dataset] -->|读取| Cleaner[Python 清洗脚本]
        Cleaner -->|1. 提取 ShortID| Meta[元数据入库 (MySQL)]
        Cleaner -->|2. 图片增强 (Albumentations)| Augment[数据增强处理]
        
        Augment -->|原图 + 20倍生成图| Split{数据集划分}
        Split -->|80%| TrainSet[yolo_dataset/train]
        Split -->|20%| ValSet[yolo_dataset/val]
    end

    subgraph "第二阶段: 模型训练 (Model Training)"
        TrainSet -->|输入| YOLO[YOLOv8-cls 训练]
        YOLO -->|迭代| Weights[best.pt 权重]
        Weights -->|导出| ONNX[best.onnx]
    end

    subgraph "第三阶段: 验证 (Validation)"
        ONNX -->|本地测试| TestScript[Python 推理脚本]
        TestScript -->|输出| Result[分类准确率报告]
    end
```

## 2. 7天极限开发周期表

### Day 1: 环境搭建与数据探查
*   **上午**: 安装 Python 环境及必要库 (`ultralytics`, `albumentations`, `opencv-python`, `pandas`)。
*   **下午**: 编写脚本遍历 `output/dataset`，统计分类数量，检查图片数量。
*   **产出**: 确认分类数量（例如 100 类），验证 YOLO 环境。

### Day 2: 核心攻坚——数据增强流水线 (最关键)
*   **全天**: 开发 `scripts/data_augment.py` (原 prepare_data.py)。
    *   **功能 1**: 从文件夹名称提取 ShortID 作为类名。
    *   **功能 2**: 集成 `Albumentations` 库，进行旋转、调色、模糊、透视变换。
    *   **功能 3**: 每张原图生成 20-30 张“变体图”。
    *   **[新增]**: 去除图片右下角水印（通过 Inpainting）。
*   **产出**: 本地 `datasets/processed/` 文件夹，填满增强后的文物图。

### Day 3: 数据集重组与验证
*   **上午**: 完成训练集/验证集划分 (80/20)。确保同一文物的原图和增强图不跨集（防止数据泄漏）。
*   **下午**: 人工抽查生成的图片，确保质量（避免过黑/变形严重）。
*   **产出**: 标准的 YOLO Classification 格式数据集，准备投喂。

### Day 4: 模型训练 (炼丹)
*   **上午**: 配置 YOLO 训练参数。
    *   **基础**: `epochs=50`, `imgsz=224`, `batch=16`。
    *   **模型选择**: 推荐使用 `yolov8s-cls.pt` (Small) 或 `yolov8m-cls.pt` (Medium) 以获得更高准确率。
*   **下午**: 开始训练。监控 Loss（损失率）和 Accuracy（准确率）。
*   **[进阶]**: 使用 `model.tune()` 进行超参数调优 (Ray Tune)，优化学习率 (`lr0`) 和 Momentum。
*   **产出**: `best.pt` 模型权重文件 (Top-1 Accuracy > 90%)。

### Day 5: 模型评估与 ONNX 导出
*   **上午**: 查看混淆矩阵 (Confusion Matrix)，分析哪些文物容易混淆。
*   **下午**: 将 `.pt` 导出为 `.onnx`，供 Java (Spring Boot) 集成使用。
*   **产出**: `model.onnx` 文件。

### Day 6: 本地模拟测试
*   **全天**: 编写 `scripts/test_inference.py`。找几张没参与训练的网图或手机拍摄图，测试 ONNX 模型的识别效果。
*   **产出**: 验证可用的模型。

### Day 7: 缓冲与文档
*   **全天**: 如果 Day 4 效果不好（过拟合），调整参数重训。整理 `labels.txt`（ShortID 到中文名的映射）。准备各类交付素材。

## 3. 核心代码实现参考

### 步骤 1: 安装依赖
```bash
pip install ultralytics albumentations opencv-python shutil
```

### 步骤 2: 数据增强脚本 (`scripts/data_augment.py`)
> 详见项目中的 `scripts/data_augment.py` 文件实现。

### 步骤 3: 开始训练 (推荐配置)
```bash
# 追求准确率：使用 yolov8s-cls.pt (Small) 或 yolov8m-cls.pt (Medium)
yolo classify train data=datasets/processed model=yolov8s-cls.pt epochs=100 imgsz=224 batch=16
```

### [进阶] 超参数调优
如果准确率不理想，可使用 Python 脚本开启调优：
```python
model = YOLO('yolov8s-cls.pt')
model.tune(data='datasets/processed', epochs=30, iterations=300, optimizer='AdamW')
```

### 步骤 4: 导出 ONNX
```bash
yolo export model=runs/classify/train/weights/best.pt format=onnx imgsz=224
```
