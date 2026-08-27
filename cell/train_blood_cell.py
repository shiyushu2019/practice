from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo11n.pt')

# 训练
results = model.train(
    data='bccd.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    project='blood_cell_detection',
    name='exp1',
    patience=20,
    pretrained=True,
    plots=True
)

print("✅ 训练完成！")

# 评估
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")