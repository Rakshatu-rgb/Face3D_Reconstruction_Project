# 🧠 Deep 3D Face Reconstruction using Neural Rendering & Morphable Models

This project focuses on reconstructing a realistic 3D model of a human face from a single 2D image using deep learning. The model leverages **ResNet**, **Graph Convolutional Networks**, and **3D Morphable Models**, with **neural rendering** techniques to achieve high-quality 3D face outputs.

## 💡 Problem Statement
Generating 3D models from 2D face images is a challenging task used in:
- Augmented Reality (AR)
- Virtual Reality (VR)
- Biometric Authentication
- Animation & Gaming

## 🛠️ Technologies Used
- **Python**
- **PyTorch**
- **ResNet Backbone**
- **Graph Convolutional Networks (GCNs)**
- **3D Morphable Models**
- **Chumpy** (parameter fitting)
- **Matplotlib / OpenCV** (for visualization)

## 📁 Project Structure
```bash
├── main/
│   └── main.py
├── models/
│   └── face_regressor/
├── checkpoints/
│   └── epoch_20.pth
├── utils/
├── test_image.jpg
├── output_3d.obj
└── README.md

