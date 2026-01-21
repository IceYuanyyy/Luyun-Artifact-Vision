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

### Day 2: 核心攻坚——细粒度识别与数据增强
*   **全天**: 开发 `scripts/data_augment.py`。
    *   **核心逻辑**: 采用 **细粒度分类 (Fine-Grained Classification)** 策略。每一个文物个体 (ShortID) 视为一个独立的类别。
    *   **[关键] 标签映射**: 在遍历文件夹时，生成 `id_to_name.json` 映射文件 (例如 `{"38f4cfb9": "战国龙纹钟"}` )，确保最终能识别出中文全名。
    *   **水印处理**: 集成去水印 Inpainting 算法。
    *   **数据增强**: 生成多角度、多光照变体，通过大量增强数据强迫模型记住纹理细节。
*   **产出**: 
    1. `datasets/processed/` (增强数据集)
    2. `datasets/id_to_name.json` (身份索引表)

### Day 3: 数据集重组与验证
*   **上午**: 完成训练集/验证集划分 (80/20)。
*   **下午**: 必须人工核对 `id_to_name.json` 的准确性，这是"精准识别"的基础。
*   **产出**: 清洗完毕的“身份数据集”。

### Day 4: 模型训练 (炼丹)
*   **策略**: 使用 **YOLOv8-Cls (Classification)** 进行实例识别。
*   **配置**:
    *   模型: `yolov8s-cls.pt` (推荐) 或 `yolov8m-cls.pt`。避免使用 Nano 模型，以保留更多特征细节。
    *   Epochs: **100** (细粒度识别需要更久收敛)。
    *   Batch: 16。
*   **目标**: Top-1 Accuracy 需达到 95% 以上。

### Day 5: 模型评估与导出
*   **上午**: 分析 Top-1 vs Top-5 准确率。对于外观极相似的文物，Top-5 包含正确答案即可视为系统有效。
*   **下午**: 导出 ONNX。

### Day 6: 识别应用开发 (GUI)
*   **全天**: 开发 `app/inference_gui.py`.
    *   **功能**: 加载 ONNX 模型 + `id_to_name.json`。
    *   **展示**: 识别后，不只显示 ID，而是直接显示 **“检测到：清代乾隆梅花诗草”**。
*   **产出**: 可交互的文物识别软件。

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
