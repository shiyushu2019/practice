from ultralytics import YOLO
import os
import random

model = YOLO('blood_cell_detection/exp1/weights/best.pt')

# 测试集图片文件夹
test_dir = 'BCCD_Dataset/test/images'

# 获取所有图片
images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

if not images:
    print("❌ 没有找到图片")
else:
    # 随机选一张
    img_name = random.choice(images)
    img_path = os.path.join(test_dir, img_name)
    
    print(f"📸 随机选择了: {img_name}")
    
    # 检测
    results = model(img_path)
    
    # 显示
    results[0].show()
    
    # 保存
    results[0].save(filename='random_result.jpg')
    print("✅ 已保存为 random_result.jpg")