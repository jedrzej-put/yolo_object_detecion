from ultralytics import YOLO
import numpy as np

import time

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
times = []
for i in range(100):
    if i < 10:
        model.predict('https://ultralytics.com/images/bus.jpg', save=True, imgsz=640, conf=0.5)
    else:
        start = time.time()
        model.predict('https://ultralytics.com/images/bus.jpg', save=True, imgsz=640, conf=0.5)
        end = time.time()
        times.append(end- start)
times = np.array(times)

print(np.mean(times))
