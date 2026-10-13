#!/usr/bin/env bash
python inf.py \
  --dataset mixed \
  --weights weights/yolo26n_mixed.pt \
  --conf 0.25 \
  --imgsz 640 \
  --iou 0.7 \
  --max-det 300 \
  --device 0
