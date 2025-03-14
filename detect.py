from ultralytics import YOLO
import torch

model = YOLO('yolov11n.pt') 

results = model("test/001.jpg")
results[0].show()