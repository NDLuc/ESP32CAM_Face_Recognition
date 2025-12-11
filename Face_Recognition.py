import cv2
import numpy as np
import requests
from ultralytics import YOLO

url = "http://192.168.1.76/capture"

# Load YOLOv8-face model
model = YOLO("yolov8n-face-lindevs.pt")

while True:
    try:
        # Lấy ảnh từ ESP32-CAM
        r = requests.get(url, timeout=2)
        img_array = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        print(img.shape)

        if img is None:
            continue

        # Chạy YOLO predict
        results = model(img, verbose=False)

        # Lấy bounding box
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Vẽ bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Hiển thị ảnh
        cv2.imshow("YOLOv8 Face Detection", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    except Exception as e:
        print("Lỗi:", e)

cv2.destroyAllWindows()
