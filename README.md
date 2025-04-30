# AI in Image and Video Processing

This project contains a series of tasks focused on image and video processing using OpenCV, MediaPipe, and deep learning models.

## Tasks Overview

### 1. Basic Image Processing
- Load and display images in different formats
- Color space conversion (Grayscale, HSV)
- Image resizing and region-of-interest (ROI) cropping

### 2. Feature Detection & Matching
- Used ORB and SIFT for keypoint detection and matching
- Match visualizations with BFMatcher

### 3. Object Detection with YOLOv3-tiny
- Used OpenCV's DNN module to load pre-trained YOLOv3-tiny
- Detected objects in static images with confidence scores and bounding boxes

### 4. Real-Time Landmark & Pose Estimation
- Pose estimation using MediaPipe
- Displayed FPS and wrote annotated video to disk

### 5. Optical Flow & Motion Analysis
- Dense (Farneback) and sparse (Lucas–Kanade) optical flow
- Comparison of motion under slow vs. fast movement

## Notes
- Videos and large data files are excluded for privacy and repository cleanliness.
- See `.gitignore` for excluded content.

## Requirements
- Python 3.10
- OpenCV
- MediaPipe
