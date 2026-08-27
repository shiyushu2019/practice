from ultralytics import YOLO
import os
import random

model_baseline = YOLO('blood_cell_detection/exp1/weights/best.pt')
model_cbam = YOLO('cf_yolov8/exp_cbam/weights/best.pt')  # ← 改成 exp_cbam

test_dir = 'BCCD_Dataset/test/images'
images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
img_name = random.choice(images)
img_path = os.path.join(test_dir, img_name)

print(f"📸 测试图片: {img_name}")

r1 = model_baseline(img_path)
r2 = model_cbam(img_path)

r1[0].save(filename='baseline_result.jpg')
r2[0].save(filename='cbam_result.jpg')

print("✅ 已保存:")
print("   - baseline_result.jpg (原始模型)")
print("   - cbam_result.jpg (CBAM模型)")

print(f"\n原始模型: {len(r1[0].boxes)} 个检测框")
print(f"CBAM模型: {len(r2[0].boxes)} 个检测框")