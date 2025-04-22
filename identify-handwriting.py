import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Open webcam (change to 0 if default camera)
cap = cv2.VideoCapture(1)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Detection parameters
CONFIDENCE_THRESHOLD = 0.3
MIN_BOX_AREA = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for display and inference consistency
    original_frame = cv2.resize(frame, (640, 640))

    # --- Preprocess for model ---
    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binarized_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Run inference on binarized frame
    results = model(binarized_frame)[0]

    found = False
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        box_area = (x2 - x1) * (y2 - y1)

        if conf > CONFIDENCE_THRESHOLD and box_area > MIN_BOX_AREA:
            found = True
            # Draw on original (non-binarized) frame
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not found:
        cv2.putText(original_frame, "No handwriting detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show output
    cv2.imshow("Handwriting Detection", original_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
