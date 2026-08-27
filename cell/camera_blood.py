import cv2
from ultralytics import YOLO

# 加载模型（注意是 exp1，不是 exp12）
model = YOLO('blood_cell_detection/exp1/weights/best.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

print("=" * 50)
print("🩸 血细胞检测系统已启动")
print("把血细胞图片放在摄像头前")
print("按 q 退出")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 摄像头打开失败")
        break
    
    # 检测
    results = model(frame, conf=0.25)
    
    # 绘制检测框
    result_frame = results[0].plot()
    
    # 显示
    cv2.imshow('Blood Cell Detection', result_frame)
    
    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 已退出")