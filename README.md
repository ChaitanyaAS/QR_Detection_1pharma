
```python
%%writefile README.md
# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**. It uses a fine-tuned YOLOv8 model to accurately detect the locations of multiple QR codes on pharmaceutical packaging. The model is designed to be robust against challenging real-world conditions such as variations in lighting, angle, and partial occlusion.

![Detection Example](https://i.imgur.com/8zR3KqV.jpeg)
*An example of the model's performance on a test image with multiple QR codes.*

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Inference (for Evaluation)](#inference-for-evaluation)
  - [Training (to Reproduce)](#training-to-reproduce)
- [Model Performance](#model-performance)

---

## 📖 Project Overview

The primary objective of this project is to solve the detection challenge by identifying all QR codes in a given image and outputting their bounding box coordinates. The solution is built using the `ultralytics` library, leveraging a pre-trained YOLOv8 model which is subsequently fine-tuned on a custom-annotated dataset.

- **Model:** YOLOv8n (nano version)
- **Framework:** PyTorch (via `ultralytics`)

---

## 📂 Repository Structure

The project is organized following the recommended hackathon structure for clarity and reproducibility:

```

├── README.md                \# This instruction file
├── requirements.txt         \# List of all Python libraries needed
├── train.py                 \# The script used to train the model from scratch
├── infer.py                 \# The primary script for running inference on test images
│
└── weights/
└── best.pt              \# The final, trained model weights for inference

````

---

## ⚙️ Setup and Installation

This section provides the exact steps required to set up the environment and run the project.

**1. Clone the Repository:**
Open your terminal and clone this GitHub repository.
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
````

**2. Create a Virtual Environment (Recommended):**
It is best practice to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies:**
All required libraries are listed in `requirements.txt`. Install them using pip.

```bash
pip install -r requirements.txt
```

-----

## 🚀 Usage

### Inference (For Evaluation)

This is the main script for the hackathon evaluation. It takes a folder of test images as input and generates a `submission.json` file with the detection results.

**To run inference, execute the following command in your terminal:**

```bash
python infer.py --input <path_to_your_test_images> --output submission.json --model_weights weights/best.pt
```

**Example:**
If you have a folder named `demo_images` in your project, the command would be:

```bash
python infer.py --input demo_images/ --output submission.json --model_weights weights/best.pt
```

This command will create the `submission.json` file in your project directory.

### Training (To Reproduce)

This script is included to show how the model was trained.

**Prerequisites:**

  - The original dataset must be unzipped and available.
  - An annotation `.zip` file in YOLO format must be provided.

**To run training:**

```bash
python train.py --annotations <path_to_labels.zip> --images <path_to_original_train_images> --project_path <folder_to_save_results>
```

-----

## 📊 Model Performance

The final model was trained for **300 epochs** on a custom-annotated dataset. It achieved the following high performance on its validation set:

  - **Precision:** 0.999
  - **Recall:** 1.0
  - **mAP50 (B):** 0.995

*(These values were retrieved from the `results.csv` of the `QR_Detection_Long_Run` training session).*

```
```
