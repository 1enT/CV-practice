import os
import cv2
from ultralytics import YOLO
import pytesseract

model = YOLO('best.pt') 

image_path = 'test_images/1/1.jpg' # Адрес на картинку
classes = ["photo", "mrz"]
tesseract_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'

output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(image_path)
h, w, _ = img.shape
cv2.imwrite(os.path.join(output_dir, f"source.jpg"), img)

results = model(image_path)[0]

boxes_text = ""
mrz_text_result = ""

for box in results.boxes:
    cls_id = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    x_center, y_center, bbox_w, bbox_h = map(float, box.xywhn[0])
    boxes_text += f"{classes[cls_id]} {x_center} {y_center} {bbox_w} {bbox_h}\n"
    crop = img[y1:y2, x1:x2]
    
    if cls_id == 0:
        cv2.imwrite(os.path.join(output_dir, "photo.jpg"), crop)
    elif cls_id == 1:
        cv2.imwrite(os.path.join(output_dir, "mrz.jpg"), crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mrz_text_result = pytesseract.image_to_string(gray, config=tesseract_config)


with open(os.path.join(output_dir, "found_boxes.txt"), "w") as f:
    f.write(boxes_text)

if mrz_text_result:
    with open(os.path.join(output_dir, "mrz_recognized.txt"), "w") as f:
        f.write(mrz_text_result.strip())
    print("Текст найден")
else:
    print("Текст не найден")