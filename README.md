# 🔒 Biometric Authentication System using Face Recognition

A robust, two-stage biometric security system that combines **YOLOv8** for face detection and **FaceNet (InceptionResnetV1)** for face recognition. This project was developed to simulate a real-world access control system, achieving high accuracy in distinguishing between authorized and unauthorized personnel.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Project Overview

This system implements a sophisticated pipeline to authenticate users via webcam feed. Unlike simple Haar-Cascade implementations, this project utilizes **Deep Learning** models for both detection and recognition, ensuring robustness against varying lighting conditions and head poses.

### Key Features
* **Hybrid Architecture**: Decouples detection (YOLOv8) from recognition (FaceNet) for modularity and performance.
* **Real-Time Processing**: Captures and processes live images from the webcam (optimized for Google Colab environments).
* **Vector Embeddings**: Converts facial features into 512-dimensional vector embeddings for precise comparison.
* **HEIC Support**: Integrated `pillow-heif` to handle high-efficiency image formats from Apple devices directly.
* **Configurable Thresholds**: Adjustable Euclidean distance tolerance to balance False Acceptance Rate (FAR) and False Rejection Rate (FRR).

---

## 🛠️ Tech Stack

* **Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Nano model)
* **Recognition**: [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) (InceptionResnetV1 pretrained on VGGFace2)
* **Image Processing**: OpenCV, Pillow (PIL), Pillow-HEIF
* **Computation**: PyTorch (CUDA supported)

---

## ⚙️ Methodology

The authentication process follows a strict 4-step pipeline:

1.  **Face Localization**: 
    The input image is passed through `yolov8n.pt`, which returns bounding boxes for all detected faces. This is superior to traditional methods as it handles occlusion and side profiles effectively.

2.  **Preprocessing & Alignment**: 
    The detected face is cropped, resized to `160x160` pixels, and normalized (RGB channels) to prepare for the neural network.

3.  **Embedding Generation**: 
    The processed face tensor is fed into the **InceptionResnetV1** model. The network outputs a unique embedding (vector) representing the facial features.

4.  **Similarity Measurement**: 
    The system calculates the **Euclidean distance** between the live embedding and the database of known users.
    * **Distance < 0.9**: Match Found → Access Granted ✅
    * **Distance > 0.9**: No Match → Access Denied 🚫

---

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have Python installed. It is recommended to run this in a GPU-accelerated environment (like Google Colab) for faster inference.

### 2. Install Dependencies
```bash
pip install ultralytics facenet-pytorch pillow-heif torch torchvision opencv-python
