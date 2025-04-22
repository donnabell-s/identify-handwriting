import cv2
import os

# Adjust these paths
images_folder = 'preprocessing/binarized/'  # Path to binarized images
labels_folder = 'preprocessing/labels/'     # Path to label files

# Only one class (you used "1" in YOLO format), so just a placeholder
class_names = ['word']  # Replace if you have more classes

for img_file in os.listdir(images_folder):
    if not img_file.endswith(('.png', '.jpg')):
        continue

    # Remove 'bin_' prefix to get the original base name for label lookup
    base_name = img_file.replace('bin_', '').rsplit('.', 1)[0]
    label_file = base_name + '.txt'
    label_path = os.path.join(labels_folder, label_file)

    img_path = os.path.join(images_folder, img_file)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"Missing label for {img_file}")
        continue

    # Read the label file and draw bounding boxes
    with open(label_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cls, x_center, y_center, box_w, box_h = map(float, line.strip().split())
            except ValueError:
                print(f"Skipping invalid line in {label_path}: {line.strip()}")
                continue

            # Convert normalized YOLO to absolute pixel coordinates
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = class_names[int(cls)] if int(cls) < len(class_names) else str(cls)
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show result
    cv2.imshow(f"Labeled Image: {img_file}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
