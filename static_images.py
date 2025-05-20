import cv2
import os
from ultralytics import YOLO

# Paths
image_folder = './preprocessing/raw_data/'
model_path = 'final.pt'

model = YOLO(model_path)

CONFIDENCE_THRESHOLD = 0.3
MIN_BOX_AREA = 100

for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, filename)
    original_img = cv2.imread(img_path)

    if original_img is None:
        print(f"Couldn't read {filename}")
        continue

    # Preprocess for model (binarized)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binarized_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Resize for inference
    resized_binarized = cv2.resize(binarized_img, (700, 640))
    resized_original = cv2.resize(original_img, (700, 640))

    # Run inference on binarized version
    results = model(resized_binarized)[0]
    print(f"{filename}: {len(results.boxes)} detections")

    found = False
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        box_area = (x2 - x1) * (y2 - y1)

        if conf > CONFIDENCE_THRESHOLD and box_area > MIN_BOX_AREA:
            found = True
            # Draw on original image
            cv2.rectangle(resized_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_original, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not found:
        cv2.putText(resized_original, "No handwriting detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show result using original (non-binarized) image
    cv2.imshow(f"Result - {filename}", resized_original)
    cv2.waitKey(0)

cv2.destroyAllWindows()
