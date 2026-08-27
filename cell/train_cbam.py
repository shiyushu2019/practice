import torch
import torch.nn as nn
from ultralytics import YOLO
from cbam import CBAM
import os

print("=" * 50)
print("🧠 CF-YOLOv8 训练 (带 CBAM 注意力)")
print("=" * 50)

# 1. 加载预训练模型
model = YOLO('yolo11n.pt')

# 2. 获取模型结构
model_instance = model.model

# 3. 在 backbone 中插入 CBAM
print("\n📌 正在注入 CBAM 注意力机制...")

# 找到 SPPF 层的位置，在它前面插入 CBAM
model_layers = list(model_instance.model)

# 找到 SPPF 层的索引
sppf_idx = None
for i, layer in enumerate(model_layers):
    if 'SPPF' in str(layer):
        sppf_idx = i
        break

if sppf_idx is not None:
    # 获取 SPPF 的输入通道数
    sppf_layer = model_layers[sppf_idx]
    in_channels = sppf_layer.cv2.conv.in_channels
    
    # 创建 CBAM 层
    cbam_layer = CBAM(in_channels, reduction=16)
    
    # 构建新的层序列：CBAM + SPPF
    from ultralytics.nn.modules import SPPF
    new_layers = nn.Sequential(
        cbam_layer,
        sppf_layer
    )
    
    # 替换
    model_instance.model[sppf_idx] = new_layers
    print(f"✅ 在位置 {sppf_idx} 插入 CBAM（通道数: {in_channels}）")
else:
    print("⚠️ 未找到 SPPF 层")

# 4. 训练
print("\n🚀 开始训练 CF-YOLOv8...")

results = model.train(
    data='bccd.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    project='cf_yolov8',
    name='exp_cbam',
    pretrained=True,
    plots=True,
    device='cpu'  # 如果 GPU 可用，改成 device=0
)

print("✅ 训练完成！")

# 5. 评估
metrics = model.val()
print("\n📊 评估结果:")
print(f"   mAP@0.5:     {metrics.box.map50:.4f}")
print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"   Precision:   {metrics.box.mp:.4f}")
print(f"   Recall:      {metrics.box.mr:.4f}")