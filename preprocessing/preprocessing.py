import cv2
import numpy as np
import os
from tqdm import tqdm

input_dir = 'raw_data/'
output_binarized_dir = 'binarized/'
output_coords_dir = 'labels/'         

os.makedirs(output_binarized_dir, exist_ok=True)
os.makedirs(output_coords_dir, exist_ok=True)

processed_images = set(
    os.path.splitext(f.replace("bin_", ""))[0]
    for f in os.listdir(output_binarized_dir)
    if f.startswith('bin_')
)

for filename in tqdm(os.listdir(input_dir)):

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        base_name = os.path.splitext(filename)[0]
        if base_name in processed_images:
            print(f"Skipping {filename} (already processed)")
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imwrite(os.path.join(output_binarized_dir, f"bin_{filename}"), binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5)) 
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        word_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 10: 
                word_boxes.append([x, y, x + w, y + h])

        print(f"{filename}: Found {len(word_boxes)} words")


        word_boxes.sort(key=lambda box: (box[1] // 50, box[0])) 

        label_data = {
            "image": filename,
            "width": img.shape[1],
            "height": img.shape[0],
            "words": []
        }

        yolo_txt_lines = []

        for box in word_boxes:
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / img.shape[1]
            y_center = ((y1 + y2) / 2) / img.shape[0]
            box_width = (x2 - x1) / img.shape[1]
            box_height = (y2 - y1) / img.shape[0]

            label_data["words"].append({
                "bbox": [x1, y1, x2, y2],
                "normalized": [x_center, y_center, box_width, box_height],
                "text": "" 
            })

            yolo_txt_lines.append(f"1 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")


        with open(os.path.join(output_coords_dir, f"{base_name}.txt"), 'w') as tf:
            tf.write('\n'.join(yolo_txt_lines))
