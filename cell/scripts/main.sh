#!/usr/bin/env bash
# 全量训练 + 验证 (数据源: bccd_preprocessed.yaml; 默认 GPU 7 / batch 32 / 100 轮)
# 各阶段需依次执行 (存在顺序依赖), 不要并行。

# 阶段1: 基线对照 YOLOv8n
python train.py --variant baseline --epochs 100 --device 7

# 阶段2: YOLOv8n + CBAM
python train.py --variant cbam --epochs 100 --device 7

# 阶段3: CF-YOLOv8 (CBAM+Focal), 文档迁移学习: 先冻结主干 10 轮, 再解冻精调 90 轮
python train.py --variant cf --epochs 100 --freeze-epochs 10 --device 7

# 阶段4: 在预处理测试集上验证所有变体
python eval.py --split test --data bccd_preprocessed.yaml --device 7 \
  --weights weights/baseline.pt,weights/cbam.pt,weights/cf.pt
