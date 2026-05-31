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
        workers=8,          # 如果仍有问题，可暂时降低 workers 或设为 0
        lr0=0.01,
        patience=50,
        save=True,
        save_period=10,
        project='runs/train',
        name='exp',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        seed=42,
        verbose=True,
    )