from ultralytics import YOLO
import os

model = YOLO('blood_cell_detection/exp1/weights/best.pt')

print("=" * 50)
print("🩸 血细胞检测系统 - 上传图片检测")
print("=" * 50)

# 输入图片路径
image_path = input("请输入图片路径（直接拖拽图片到终端）: ").strip().strip('"')

if not os.path.exists(image_path):
    print(f"❌ 找不到文件: {image_path}")
    exit()

print(f"📸 正在检测: {image_path}")

# 检测
results = model(image_path)

# 显示结果
results[0].show()

# 保存结果
output_name = 'detect_result.jpg'
results[0].save(filename=output_name)

# 打印检测结果
boxes = results[0].boxes
if len(boxes) > 0:
    print(f"\n✅ 检测到 {len(boxes)} 个细胞:")
    counts = {}
    for box in boxes:
        cls = results[0].names[int(box.cls)]
        counts[cls] = counts.get(cls, 0) + 1
    for cls, count in counts.items():
        print(f"   - {cls}: {count} 个")
else:
    print("❌ 未检测到血细胞")

print(f"\n💾 结果已保存为: {output_name}")