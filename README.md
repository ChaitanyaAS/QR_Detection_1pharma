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

The project is organized following the recommended hackathon structure:

multiqr-hackathon/
│
├── README.md                # This file: Setup and usage instructions
├── requirements.txt         # Python dependencies for the project
├── train.py                 # Script for training the model
├── infer.py                 # Script for running inference and generating the submission file
│
└── weights/
└── best.pt              # The final trained model weights for inference


---

## Setup and Installation

To set up the project environment, please follow these steps. This project was developed and tested using Python 3.10.

**1. Clone the Repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create a Virtual Environment (Recommended):
It is recommended to use a virtual environment to manage project dependencies.

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependencies:
All required libraries are listed in the requirements.txt file. Install them using pip:

Bash

pip install -r requirements.txt
Usage
The project includes two primary scripts: train.py for training a new model and infer.py for generating predictions.

Training
The train.py script fine-tunes a YOLOv8 model on a custom dataset.

Prerequisites:

An annotation .zip file in YOLO format, which must contain both the .jpg image files and their corresponding .txt label files.

The original dataset's train_images folder must be available and correctly path-referenced if the annotation zip does not contain images.

To run the training script:

Place your prepared annotation .zip file in the root of the project directory.

Modify the zip_filename variable inside the train.py script to match the name of your file.

Execute the script from the command line:

Bash

python train.py
The final trained model (best.pt) and other training artifacts will be saved to a new directory.

Inference
The infer.py script is used to generate the final submission_detection_1.json file for the hackathon. It requires a folder of test images and the path to the trained model weights.

To generate the submission file, run the following command:
Make sure to replace <path_to_test_images> with the actual path to the folder containing the test images.

Bash

python infer.py --input <path_to_test_images> --output submission.json --model_weights weights/best.pt
Example:

Bash

python infer.py --input data/test_images/ --output submission.json --model_weights weights/best.pt
This command will process all images in the specified input directory and create the submission.json file in the project's root.

Model Performance
The model was trained for 300 epochs on a custom-annotated dataset of XX images (please replace XX with the number of images you labeled). The final model achieved the following performance on its validation set:

Precision: 0.XXX

Recall: 0.XXX

mAP50 (B): 0.XXX
