from collections import defaultdict

import cv2
import numpy as np


import copy

from ultralytics import YOLO
from kalman_filter import KalmanFilterXYWH

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "/ros2_ws/pexels_videos_2053100.mp4"

cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

my_track_id = 15
founded = False
my_obj_index = None
obj_box = None
first_detected = False
kalman_filter_xywh = KalmanFilterXYWH()
mean, covariance = None, None

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if my_track_id in track_ids:
            
            founded = True
            my_obj_index = track_ids.index(my_track_id)
            obj_box = boxes[my_obj_index]
        else:
            founded = False

        if founded:
            if not first_detected:
                mean, covariance = kalman_filter_xywh.initiate(obj_box)
                first_detected = True
        
        if first_detected:
            mean, covariance = kalman_filter_xywh.predict(mean, covariance)

        if founded:
            mean, covariance = kalman_filter_xywh.update(mean, covariance, obj_box)

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )
        
        if first_detected:
            mean_25, covariance_25 = copy.deepcopy(mean), copy.deepcopy(covariance)
            for i in range(25):
                mean_25, covariance_25 = kalman_filter_xywh.predict(mean_25, covariance_25)

            start_point = (int(mean[0]), int(mean[1]))
            end_point = (int(mean_25[0]), int(mean_25[1]))
            color = (0, 255, 0)  
  
            # Line thickness of 9 px  
            thickness = 9
            annotated_frame = cv2.arrowedLine(annotated_frame, start_point, end_point, 
                                     color, thickness)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()