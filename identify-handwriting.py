import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Open webcam (use 1 if USB webcam is second camera, else try 0)
cap = cv2.VideoCapture(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Parameters
CONFIDENCE_THRESHOLD = 0.3
MIN_BOX_AREA = 100  # Filter tiny boxes (like blank spaces)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize if needed (YOLOv8 handles different sizes well, optional)
    resized = cv2.resize(frame, (640, 640))

    # Run inference
    results = model(resized)[0]

    found = False
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        box_area = (x2 - x1) * (y2 - y1)

        if conf > CONFIDENCE_THRESHOLD and box_area > MIN_BOX_AREA:
            found = True
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put label
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not found:
        cv2.putText(frame, "No handwriting detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show result
    cv2.imshow("Handwriting Detection", frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
