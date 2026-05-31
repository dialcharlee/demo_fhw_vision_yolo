from ultralytics import YOLO

model = YOLO("runs/detect/runs/train/exp/weights/best.pt")

model.export(format="onnx")