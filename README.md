# Object-Tracking-with-YOLO
This repository provides an implementation for **object detection** and **tracking** using the YOLOv3 and DeepSort.

## Build Docker Image
From the root directory of the project:
```bash
docker build -t yolov3_image .
```

## Run the Container
```bash
docker run -it yolov3_image
```

## Run Inference
```bash
/usr/bin/python3 main.py
```

## Results
Video of the tracking result saved in media file
