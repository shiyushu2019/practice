#!/usr/bin/env bash
python eval.py \
  --split test \
  --device 7 \
  --dataset mixed \
  --imgsz 640 \
  --iou 0.7
