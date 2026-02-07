# 👻 Universal Ghost Patch V2: Saliency-Guided Adversarial IoT

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNet%2FResNet-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-C51A4A?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Scientific Research Project (Phase 2)**
> *Real-time Physical Adversarial Attacks on Edge Devices using Attention-Guided Mechanisms.*

## 📖 Overview

**Universal Ghost Patch V2** is a framework for deploying physical adversarial attacks on edge devices (like Raspberry Pi 4). Unlike traditional methods that rely on random patch placement, this project introduces a **Deterministic Saliency-Guided Mechanism**.

By analyzing the model's gradients in real-time (White-box approach), the system dynamically identifies the "most salient" regions of a video frame—where the AI focuses its attention—and positions the adversarial patch exactly there. This maximizes the **Attack Success Rate (ASR)** while minimizing the occlusion area.

---

## ⚡ Key Innovations: Phase 1 vs. Phase 2

| Feature | Phase 1 (Legacy) | 🚀 Phase 2 (Current - IoT) |
| :--- | :--- | :--- |
| **Attack Strategy** | Genetic Algorithm (Stochastic) | **Saliency-Guided Gradient (Deterministic)** |
| **Patch Placement** | Random or Center-fixed | **Dynamic Max-Attention Targeting** |
| **Optimization** | Blind "Black-box" Search | **White-box Gradient Heatmap** $\nabla_x \mathcal{L}$ |
| **Architecture** | Monolithic Script | **Modular MVC (Model-View-Controller)** |
| **Deployment** | Local Python Execution | **Dockerized Microservice** |
| **Interface** | CLI / Static Images | **Real-time Web Dashboard (Flask)** |

---

## 🧠 Core Logic: How It Works

The core algorithm sits inside `src/core.py` and executes the following pipeline for every frame:

1.  **Ensemble Inference**: The input frame $x$ is passed through an ensemble of **MobileNetV2** and **ResNet50**.
2.  **Sensitivity Analysis**: We compute the gradient of the loss function with respect to the input image:
    $$S(x) = \left| \frac{\partial \mathcal{L}}{\partial x} \right|$$
3.  **Noise Smoothing**: A Gaussian Blur ($K=15, \sigma=5$) is applied to $S(x)$ to aggregate pixel-level noise into coherent **Regions of Interest (ROI)**.
4.  **Target Localization**: The system identifies coordinates $(u^*, v^*) = \arg\max S_{blurred}(x)$ and snaps the adversarial patch center to this location.
5.  **EOT Integration**: Expectation Over Transformation (EOT) is applied to ensure robustness against lighting and angle changes.

---

## 🏗 Project Structure

Refactored for modularity and Docker support:

```text
UniversalGhostPatch/
├── Dockerfile              # Container configuration (ARM64 optimized)
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── src/                    # [CORE] Application Logic
│   ├── __init__.py
│   ├── core.py             # Saliency calculation & Patch application
│   ├── models.py           # Model loaders (MobileNet/ResNet)
│   └── utils.py            # EOT transformations & Image processing
├── web/                    # [WEB] Interface
│   ├── app.py              # Flask Server entry point
│   └── templates/          # HTML Dashboard
├── tool/                   # [TOOLS] Standalone scripts
│   ├── train.py            # Offline patch training (GA)
│   └── test_cam.py         # Local camera testing
├── docs/                   # Documentation & Reports
└── data/                   # Dataset & Assets
```

---

## 🚀 Deployment Guide

### Prerequisites
*   **Hardware**: Raspberry Pi 4 (4GB RAM recommended) or any Linux/Windows PC.
*   **Camera**: USB Webcam (V4L2 compatible).
*   **Software**: Docker Engine & Docker Compose.

### Method 1: Docker (Recommended)
This method automatically handles all dependencies, including system libraries for OpenCV on ARM64.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/UniversalGhostPatch.git
cd UniversalGhostPatch

# 2. Build and Run the container
# This maps the USB camera (/dev/video0) to the container
docker-compose up --build -d

# 3. View logs (optional)
docker-compose logs -f
```

### Method 2: Manual Installation (Python)
Suitable for development/debugging.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Web Server
# Make sure your PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)
python web/app.py
```

---

## 🎮 User Interface & Usage

Once running, access the dashboard at:
👉 `http://localhost:5000` (or `http://<RASPBERRY_PI_IP>:5000`)

**Dashboard Controls:**
*   **Live Feed**: Displays the clean MJPEG stream from the webcam.
*   **Status Panel**: Shows current FPS, Model Prediction, and Confidence Score.
*   **"ACTIVATE GHOST" Button**:
    *   Toggles the Saliency Attack Mode.
    *   The system will start visualizing the attack in real-time.
    *   Watch the patch automatically "jump" to the most recognized features (e.g., faces) to force a misclassification.

---

## 📊 Performance Metrics (Raspberry Pi 4)

| Metric | Clean Stream | Attack ON (Saliency) |
| :--- | :--- | :--- |
| **FPS** | ~24 FPS | ~5-8 FPS |
| **Latency** | < 50ms | ~150ms |
| **CPU Load** | 15% | 85% (4 Cores) |
| **ASR** | N/A | 81.2% (Target: Toaster) |

---

## 👨‍💻 Authors & Acknowledgments
*   **Reikage**: Core Algorithm (Saliency Logic), Genetic Optimization.
*   **BaoZ**: IoT Architecture, Dockerization, Web Interface.

*Project conducted under the Scientific Research Program (NCKH) 2024-2025.*

## ⚖️ Disclaimer
*This project is for educational and research purposes only. The authors are not responsible for any misuse of the provided code.*
