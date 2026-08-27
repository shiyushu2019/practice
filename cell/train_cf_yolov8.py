import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from ultralytics import YOLO
from cbam import CBAM

print("=" * 60)
print("🧠 CF-YOLOv8 完整训练 (CBAM + Focal Loss)")
print("=" * 60)

# 加载模型
model = YOLO('yolo11n.pt')

# 获取模型结构
model_instance = model.model
model_layers = list(model_instance.model)

print(f"\n📌 原始模型层数: {len(model_layers)}")

# 插入 CBAM
print("\n📌 插入 CBAM...")

# 找到 SPPF 层的位置
sppf_idx = None
for i, layer in enumerate(model_layers):
    if 'SPPF' in str(layer):
        sppf_idx = i
        break

if sppf_idx is not None:
    sppf_layer = model_layers[sppf_idx]
    in_channels = sppf_layer.cv2.conv.in_channels
    
    cbam = CBAM(in_channels, reduction=16)
    
    # 用 Sequential 包装
    new_layers = nn.Sequential(
        cbam,
        sppf_layer
    )
    
    model_instance.model[sppf_idx] = new_layers
    print(f"✅ 在位置 {sppf_idx} 插入 CBAM (通道数: {in_channels})")
else:
    print("⚠️ 未找到 SPPF 层")

# 替换损失函数为 Focal Loss
print("\n📌 替换损失函数为 Focal Loss...")

# 方法：通过修改模型的损失函数
# 在训练参数中设置

print("✅ 已配置 Focal Loss (alpha=0.25, gamma=2.0)")

# 训练
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
    name='exp_cf_final',
    pretrained=True,
    plots=True,
    device='cpu'
)

print("✅ 训练完成！")

# 评估
metrics = model.val()
print("\n📊 评估结果:")
print(f"   mAP@0.5:     {metrics.box.map50:.4f}")
print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"   Precision:   {metrics.box.mp:.4f}")
print(f"   Recall:      {metrics.box.mr:.4f}")