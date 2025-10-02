Here’s a clean, professional version of your README without emojis and with consistent heading styles:

```markdown
# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**.  
It uses a fine-tuned YOLOv8 model to accurately detect the location of multiple QR codes on pharmaceutical packaging, even in challenging conditions such as tilted, blurry, or partially occluded images.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Repository Structure](#repository-structure)  
- [Setup and Installation](#setup-and-installation)  
- [Usage](#usage)  
  - [Training](#training)  
  - [Inference](#inference)  
- [Model Performance](#model-performance)  

---

## Project Overview

The primary goal of this project is to solve the detection challenge by identifying all QR codes in a given image and outputting their bounding box coordinates.  
The solution is built using the `ultralytics` library, leveraging a pre-trained YOLOv8 model which is then fine-tuned on a custom-annotated dataset.

- **Model:** YOLOv8n (nano version)  
- **Framework:** PyTorch (via `ultralytics`)  
- **Training Data:** A custom-annotated subset of the provided training images  

---

## Repository Structure

The project follows the recommended hackathon structure:

```

multiqr-hackathon/
│
├── README.md                # Setup and usage instructions
├── requirements.txt         # Python dependencies
├── train.py                 # Script for training the model
├── infer.py                 # Script for inference and submission file generation
│
└── weights/
└── best.pt              # Final trained model weights

````

---

## Setup and Installation

Follow these steps to set up the project.  
This project was developed using **Python 3.10**.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

All required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## Usage

This project is divided into two main scripts:

* `train.py` for training the model
* `infer.py` for generating the submission file

### Training

The `train.py` script trains the YOLOv8 model using a `.zip` file containing annotated images and labels.

**Prerequisites:**

* An annotation `.zip` file in YOLO format containing both `.jpg` images and corresponding `.txt` label files
* The dataset's `train_images` folder available

**Steps:**

1. Place your annotation `.zip` file in the root of the project directory.
2. Update the `zip_filename` inside the `train.py` script with the name of your file.
3. Run the training script:

```bash
python train.py
```

The trained model (`best.pt`) will be saved in a new folder in your Google Drive under `MultiQR_Hackathon_Project/`.

---

### Inference

The `infer.py` script generates the final `submission_detection_1.json` file for the hackathon.
It takes a folder of images and the trained model weights as input.

**Example Command:**

```bash
python infer.py --input data/test_images/ --output submission_detection_1.json --model_weights weights/best.pt
```

This will create the `submission_detection_1.json` file in the project root directory.

---

## Model Performance

The model was trained for 100 epochs on a custom-annotated dataset of **XX images** (replace XX with the actual number).
The final model achieved the following performance on the validation set:

* **Precision:** 0.XXX
* **Recall:** 0.XXX
* **mAP50 (Bounding Boxes):** 0.XXX

```

---

Would you like me to also **add a section for "Future Improvements"** (e.g., better augmentation, larger model, deployment options), or should I keep it strictly as per hackathon submission guidelines?
```
