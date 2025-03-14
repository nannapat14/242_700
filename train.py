from ultralytics import YOLO
import torch

model = YOLO("yolov11n.pt")

#train the model
train_results = model.train(
    data="datasets/data.yaml", #path to dataset YAML
    epochs=5, #number of training epochs
    imgsz=640, #image size
    device="cpu"
)
#Evluate the model
metrics = model.val()