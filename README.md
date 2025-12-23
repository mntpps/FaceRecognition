# 🔒 Biometric Authentication System using Face Recognition

A robust, two-stage biometric security system that combines **YOLOv8** for face detection and **FaceNet (InceptionResnetV1)** for face recognition. This project was developed to simulate a real-world access control system.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YourUsername]/[YourRepoName]/blob/main/FaceRecognition.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)

## ⚠️ Recommended Environment: Google Colab
**This project is optimized for Google Colab.** Because the system utilizes deep learning models (YOLOv8 & InceptionResnet), a GPU environment is highly recommended for smooth performance.
* **Click the "Open in Colab" badge above** to launch the notebook directly.
* Ensure you change the runtime type to **T4 GPU** (Runtime > Change runtime type > T4 GPU).

---

## 📖 Project Overview

This system implements a sophisticated pipeline to authenticate users via webcam feed. Unlike simple Haar-Cascade implementations, this project utilizes **Deep Learning** models for both detection and recognition, ensuring robustness against varying lighting conditions and head poses.

### Key Features
* **Hybrid Architecture**: Decouples detection (YOLOv8) from recognition (FaceNet).
* **Real-Time Processing**: Captures and processes live images from the webcam.
* **Vector Embeddings**: Converts facial features into 512-dimensional vector embeddings.
* **HEIC Support**: Integrated `pillow-heif` to handle Apple image formats directly.

---

## 🛠️ Tech Stack

* **Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Nano model)
* **Recognition**: [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) (InceptionResnetV1)
* **Image Processing**: OpenCV, Pillow (PIL), Pillow-HEIF
* **Computation**: PyTorch (CUDA supported)

---

## ⚙️ Methodology

The authentication process follows a strict 4-step pipeline:

1.  **Face Localization**: Passes input through `y
