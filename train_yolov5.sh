#!/usr/bin/env bash
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data yolo_dataset/data.yaml --weights yolov5s.pt --name smart_kitchen_exp
