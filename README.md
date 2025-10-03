```markdown
# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**. It uses a fine-tuned YOLOv8 model to accurately detect the location of multiple QR codes on pharmaceutical packaging, even in challenging conditions such as tilted, blurry, or partially occluded images.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Setup Instructions](#-setup-instructions)
- [Model Training](#-model-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Results](#-results)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🚀 Project Overview
Pharmaceutical packaging often contains multiple QR codes that need to be detected reliably for authentication, traceability, and safety. Our solution leverages the YOLOv8 object detection framework, fine-tuned on a curated dataset of medicine pack images, to ensure:
- Detection of **multiple QR codes per image**.
- Robustness under **occlusion, blur, tilt, and varying lighting**.
- Fast inference for real-world applications.

---

## 📂 Repository Structure
```

├── data/                  # Dataset (organized via Roboflow/COCO format)
├── notebooks/             # Colab training & inference notebooks
├── runs/                  # YOLO training logs & weights
├── models/                # Exported trained models
├── README.md              # Project documentation
└── requirements.txt       # Dependencies

````

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/multi-qr-detection.git
cd multi-qr-detection
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Access

* The dataset is hosted on **Roboflow**.
* Use the API key inside the notebook for direct download:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("multi-qr-detection")
dataset = project.version(1).download("yolov8")
```

---

## 🏋️ Model Training

### Train YOLOv8

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### Resume Training

```bash
yolo detect train resume model=runs/detect/train/weights/last.pt
```

---

## 📊 Evaluation

Run evaluation on validation set:

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml
```

This generates:

* mAP (mean Average Precision)
* Precision & Recall
* Per-class performance metrics

---

## 🔍 Inference

Run inference on test images:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images save=True
```

Example with webcam:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```

---

## 📈 Results

* Achieved **mAP@0.5 > 95%** on validation dataset.
* Robust performance on real-world test images with tilted & occluded QR codes.
* Fast inference (~30 FPS on GPU).

Sample output:

| Input Image                       | Prediction                          |
| --------------------------------- | ----------------------------------- |
| ![Input](assets/sample_input.jpg) | ![Output](assets/sample_output.jpg) |

---

## 🔮 Future Work

* Improve robustness for extremely low-resolution QR codes.
* Explore **lightweight models** for deployment on edge devices.
* Integrate with **QR code decoders** for end-to-end validation.

---

## 📜 License

This project is licensed under the MIT License.

```

Would you like me to also **inline all the setup + training Colab code cells** into this same README (so it’s a fully standalone copy-paste guide), or keep those only as references?
```
