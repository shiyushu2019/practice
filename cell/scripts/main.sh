#!/usr/bin/env bash
# 全量训练 + 验证 + 推理 (数据集: 默认混合 BCCD + TXL-PBC; 默认 GPU 1 / batch 16 / 100 轮)
# 消融五组: baseline / +Focal Loss / +CBAM / +DAMS-Net / CF-YOLOv8; 各阶段需依次执行, 不要并行。

# 阶段1: 基线对照 YOLOv8n
python train.py --variant baseline --epochs 100 --dataset mixed --device 3

# 阶段2: YOLOv8n + Focal Loss (消融)
python train.py --variant focal --epochs 100 --dataset mixed --device 3

# 阶段3: YOLOv8n + CBAM
python train.py --variant cbam --epochs 100 --dataset mixed --device 3

# 阶段4: YOLOv8n + DAMS-Net (消融)
python train.py --variant dams --epochs 100 --dataset mixed --device 3

# 阶段5: 完整 CF-YOLOv8 (CBAM + Focal + DAMS-Net), 按文档配置 (Adam/lr0=0.001/batch16/patience10/先冻结后解冻)
python train.py --variant cf --dams --epochs 100 --dataset mixed --device 3 \
  --gamma 2.0 --alpha 0.25 --batch 16 --freeze-epochs 10

# 阶段6: 在混合测试集上验证所有变体
python eval.py --split test --dataset mixed --device 3 \
  --weights weights/baseline.pt,weights/focal.pt,weights/cbam.pt,weights/dams.pt,weights/cf_dams.pt

# 阶段7: 在混合测试集上推理 (CF-YOLOv8) + FPS 基准 + 保存可视化结果图
python inf.py --dataset mixed --weights weights/cf_dams.pt --device 3 --benchmark --save
