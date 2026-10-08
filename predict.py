from ultralytics import YOLO

model = YOLO('best.pt')

for i in range(1, 24):
	results = model.predict(
		source=f"test/2/{i}.jpg",
		conf=0.1,
		save=True
	)