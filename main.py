import cv2
import numpy as np
import os


# -------------------------
# Configurations
# -------------------------
CFG_PATH = "data/raw/yolov3-training.cfg"
WEIGHTS_PATH = "model/yolov3-training.weights"
VIDEO_PATH = "media/video.mp4"
OUTPUT_PATH = "media/detection_tracking_output.mp4"

CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.5

# -------------------------
# Load YOLOv3
# -------------------------
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)

# Prefer GPU if available
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("[INFO] Using GPU for computation.")
except:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] Using CPU for computation.")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# -------------------------
# Initialize Tracker
# -------------------------
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

# -------------------------
# Video setup
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out_video = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_id = 0

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Detection ----
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []
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

    # ---- Tracking ----
    detections_np = np.array([boxes[i] + [confidences[i]] for i in range(len(boxes))])
    tracked_objects = tracker.update(detections_np)

    # ---- Drawing results ----
    for x1, y1, x2, y2, track_id in tracked_objects:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out_video.write(frame)
    frame_id += 1

cap.release()
out_video.release()
print(f"[INFO] Detection + tracking video saved to {OUTPUT_PATH}")
