from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO(r"yolo11n.pt")

    # 开始训练
    results = model.train(
        data='robot.v2i.yolov11/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        lr0=0.01,
        project='runs/train',
    )