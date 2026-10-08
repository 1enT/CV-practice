from ultralytics import YOLO

model = YOLO('yolo26n.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu',
    patience=10,
    pretrained=True,
    workers=4,
    amp=False,
    plots=True
)