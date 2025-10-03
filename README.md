# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**. It uses a fine-tuned YOLOv8 model to accurately detect the locations of multiple QR codes on pharmaceutical packaging. The model is designed to be robust against challenging real-world conditions such as variations in lighting, angle, and partial occlusion.

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

The primary objective of this project is to solve the detection challenge by identifying all QR codes in a given image and outputting their bounding box coordinates. The solution is built using the `ultralytics` library, leveraging a pre-trained YOLOv8 model which is subsequently fine-tuned on a custom-annotated dataset derived from the competition's training images.

- **Model:** YOLOv8n (nano)  
- **Framework:** PyTorch (via `ultralytics`)  
- **Training Data:** A custom-annotated subset of the provided training images.  

---

## Repository Structure

multiqr-hackathon/  
│  
├── README.md                # Setup and usage instructions  
├── requirements.txt         # Python dependencies for the project  
├── train.py                 # Script for training the model  
├── infer.py                 # Script for running inference and generating the submission file  
│  
└── weights/  
  └── best.pt              # Final trained model weights for inference  

---

## Setup and Installation

This project was developed and tested with **Python 3.10**.  

1. **Clone the Repository:**  
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  

2. **Create a Virtual Environment (Recommended):**  
python -m venv venv  
source venv/bin/activate   # On Windows: venv\Scripts\activate  

3. **Install Dependencies:**  
pip install -r requirements.txt  

---

## Usage

The project includes two primary scripts:  
- train.py → for training a new model  
- infer.py → for generating predictions  

### Training

The train.py script fine-tunes a YOLOv8 model on a custom dataset.  

**Prerequisites:**  
- An annotation .zip file in YOLO format, containing both .jpg image files and their corresponding .txt label files.  
- If the .zip does not contain images, the original dataset’s train_images/ folder must be available and referenced correctly.  

**Steps:**  
1. Place your prepared annotation .zip file in the root of the project directory.  
2. Modify the zip_filename variable inside the train.py script to match the name of your file.  
3. Run training:  
python train.py  

The final trained model (best.pt) and training artifacts will be saved to a new directory.  

---

### Inference

The infer.py script generates the final submission.json file required for the hackathon. It requires:  
- A folder of test images  
- Path to the trained model weights  

**Run Inference:**  
python infer.py --input <path_to_test_images> --output submission.json --model_weights weights/best.pt  

**Example:**  
python infer.py --input data/test_images/ --output submission.json --model_weights weights/best.pt  

This will process all images in the input directory and create the submission.json file in the project root.  

---

## Model Performance

The model was trained for 300 epochs on a custom-annotated dataset of XX images (replace XX with the number of labeled images).  
Final validation performance:  

- Precision: 0.XXX  
- Recall: 0.XXX  
- mAP50 (B): 0.XXX  
