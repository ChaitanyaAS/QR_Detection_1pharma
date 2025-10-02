%%writefile README.md
# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**. It uses a fine-tuned YOLOv8 model to accurately detect the location of multiple QR codes on pharmaceutical packaging, even in challenging conditions such as tilted, blurry, or partially occluded images.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
  - [Training](#-training)
  - [Inference](#-inference)
- [Model Performance](#-model-performance)

---

## 📖 Project Overview

The primary goal of this project is to solve the main detection challenge by identifying all QR codes in a given image and outputting their bounding box coordinates. The solution is built using the `ultralytics` library, leveraging a pre-trained YOLOv8 model which is then fine-tuned on a custom-annotated dataset.

* **Model:** YOLOv8n (nano version)
* **Framework:** PyTorch (via `ultralytics`)
* **Training Data:** A custom-annotated subset of the provided training images.



---

## 📂 Repository Structure

The project is organized according to the recommended hackathon structure:
multiqr-hackathon/
│
├── README.md                # This file: Setup & usage instructions
├── requirements.txt         # Python dependencies
├── train.py                 # Script for training the model
├── infer.py                 # Script for running inference and generating submission.json
│
└── weights/
└── best.pt              # The final trained model weights


---

## ⚙️ Setup and Installation

To get the project running, follow these steps. This project was developed using Python 3.10.

**1. Clone the Repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create a Virtual Environment (Recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependencies:
All required libraries are listed in the requirements.txt file.

Bash

pip install -r requirements.txt
🚀 Usage
This project is divided into two main scripts: train.py for training the model and infer.py for generating the submission file.

▶️ Training
The train.py script is designed to take a .zip file containing annotated images and labels and train the YOLOv8 model.

Prerequisites:

An annotation .zip file in YOLO format, containing both the .jpg images and their corresponding .txt label files.

The original dataset's train_images folder available.

To run the training script:

Place your annotation .zip file in the root of the project directory.

Update the zip_filename inside the train.py script with the name of your file.

Run the script from the command line:

Bash

python train.py
The trained model (best.pt) will be saved in a new folder in your Google Drive under MultiQR_Hackathon_Project/.

▶️ Inference
The infer.py script is used to generate the final submission_detection_1.json file for the hackathon. It takes a folder of images and the trained model weights as input.

To generate the submission file, run the following command:

Make sure to replace <path_to_test_images> with the path to the folder containing the test images.

Bash

python infer.py --input <path_to_test_images> --output submission.json --model_weights weights/best.pt
For example:

Bash

python infer.py --input data/test_images/ --output submission_detection_1.json --model_weights weights/best.pt
This will create the submission_detection_1.json file in the project's root directory.

📊 Model Performance
The model was trained for 100 epochs on a custom-annotated dataset of XX images (replace XX with the number of images you labeled). The final model achieved the following performance on its validation set:

Precision: 0.XXX

Recall: 0.XXX

mAP50(B): 0.XXX
