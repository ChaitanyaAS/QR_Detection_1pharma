
# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**. It uses a fine-tuned YOLOv8 model to accurately detect the locations of multiple QR codes on pharmaceutical packaging. The model is robust against challenging real-world conditions such as variations in lighting, angle, and partial occlusion.

Additionally, this repository includes a **bonus challenge solution**: decoding the QR content into Binary Code.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Step-by-Step Guide for Evaluation](#step-by-step-guide-for-evaluation)
  - [1. Setup and Installation](#1-setup-and-installation)
  - [2. Running Inference (Official Evaluation)](#2-running-inference-official-evaluation)
  - [3. Running the Bonus QR Decoder](#3-running-the-bonus-qr-decoder)
- [Additional Scripts](#additional-scripts)
  - [Visual Demonstration](#visual-demonstration)
  - [Reproducing the Training Process](#reproducing-the-training-process)
- [Model Performance](#model-performance)

---

## 📖 Project Overview

The primary objective of this project is to solve the detection challenge by identifying all QR codes in a given image and outputting their bounding box coordinates.  

The **bonus challenge** extends this by:

1. **Decoding the QR Content** – converting the QR code black/white modules into actual text or numbers (e.g., "B12345").  

- **Detection Model:** YOLOv8n (nano version)  
- **Framework:** PyTorch (via `ultralytics`)  
- **Bonus Decoder:** Custom Python script to extract and classify QR contents.

---

##  Repository Structure

```

├── README.md                       # This instruction file
├── requirements.txt                # List of all Python libraries needed
├── train.py                        # Script to train the YOLO model from scratch
├── infer.py                         # Primary script for detection inference
├── bonus_infer.py                  # Bonus script: QR content decoding & classification
├── visual_test.py                  # Optional script for visual demonstration
│
└── weights/
└── best.pt                     # Final trained model weights

````

---

##  Step-by-Step Guide for Evaluation

### 1. Setup and Installation

**1.1. Clone the Repository**

```bash
git clone https://github.com/ChaitanyaAS/QR_Detection_1pharma.git    
cd QR_Detection_1Pharma
````

**1.2. Create a Virtual Environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**1.3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### 2. Running Inference (Official Evaluation)

The `infer.py` script is the primary tool for evaluation. It processes a folder of images and generates a `submission.json` file with bounding box coordinates.

```bash
python infer.py --input <path_to_your_test_images> --output submission.json --model_weights weights/best.pt
```

**Example:**

```bash
python infer.py --input demo_images/ --output submission.json --model_weights weights/best.pt
```

---

### 3. Running the Bonus QR Decoder

The `bonus_infer.py` script **decodes the QR content** and **classifies it** after detection.

**Single Image Example:**

```bash
python bonus_infer.py --input "demo_images/sample_image.png" --output "submission_decoding.json" --model_weights "weights/best.pt"
```

**Folder of Images Example:**

```bash
python bonus_infer.py --input "demo_images/" --output "submission_decoding.json" --model_weights "weights/best.pt"
```

**What this script does:**

1. Detects all QR codes in the image(s) using YOLOv8.
2. Crops and processes each QR code.
3. Converts the QR modules into binary grids.
4. Decodes the QR content (alphanumeric text).
5. Saves the results in a JSON file with all decoded texts and classifications.

---

##  Additional Scripts

### Visual Demonstration

Displays detected QR codes with bounding boxes.

```bash
python visual_test.py --image "demo_images/your_image_name.jpg" --model_weights "weights/best.pt"
```

### Reproducing the Training Process

To retrain the YOLOv8 model:

```bash
python train.py --annotations <path_to_labels.zip> --images <path_to_train_images> --project_path <folder_to_save_results>
```

---

##  Model Performance

Trained for **300 epochs** on a custom-annotated dataset:

* **Precision:** 0.999
* **Recall:** 1.0
* **mAP50 (B):** 0.995

The bonus decoder has been tested to correctly extract and classify QR content in all validation images.

```

