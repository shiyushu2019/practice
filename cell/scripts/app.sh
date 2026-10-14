#!/usr/bin/env bash
python app.py \
  --host 0.0.0.0 \
  --port 5000 \
  --weights weights/cf.pt \
  --conf 0.25
