#!/usr/bin/env bash
python app.py \
  --host 0.0.0.0 \
  --port 5000 \
  --weights weights/yolo26n_mixed.pt \
  --conf 0.25
