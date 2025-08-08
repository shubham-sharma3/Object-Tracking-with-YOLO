import cv2
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deepsort_tracker import Tracker
from tqdm import tqdm

# Configurations
CFG_PATH = "model/yolov3_training.cfg"
WEIGHTS_PATH = "model/yolov3_training.weights"
VIDEO_PATH = "media/video.mp4"
OUTPUT_PATH = "media/detection_tracking_output.mp4"

CONF_THRESHOLD = 0.8
NMS_THRESHOLD = 0.3

# Load YOLOv3
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize Tracker
tracker = DeepSort(max_age=30,nms_max_overlap=0.3, n_init=5, max_iou_distance=0.4) 

# Video setup
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out_video = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_id = 0

# Main loop
# Main loop with progress bar
progress_bar = tqdm(total=total_frames, desc="Processing video", unit="frames")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Detection ----
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (608, 608), (0, 0, 0), True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, x+w, y+h])  # SORT format
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # ---- Tracking ----
    # detections_np = np.array([boxes[i] + [confidences[i]] for i in range(len(boxes))])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    dets = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1,y1,x2,y2 = boxes[i]
            conf = confidences[i]
            cls_id = class_ids[i]
            dets.append([[x1, y1, x2, y2], conf, cls_id])

    
    tracks = tracker.update_tracks(dets, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out_video.write(frame)
    frame_id += 1

    # Update progress bar
    progress_bar.update(1)
    progress_bar.set_postfix({
        'Frame': frame_id,
        'Detections': len(dets) if 'dets' in locals() else 0
    })

progress_bar.close()
cap.release()
out_video.release()
print(f"[INFO] Detection + tracking video saved to {OUTPUT_PATH}")
