#!/usr/bin/env bash
python eval.py \
  --split test \
  --device 7 \
  --data dataset/bccd.yaml \
  --imgsz 640 \
  --iou 0.7
