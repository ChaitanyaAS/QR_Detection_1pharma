
# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**. It uses a fine-tuned YOLOv8 model to accurately detect the locations of multiple QR codes on pharmaceutical packaging. The model is designed to be robust against challenging real-world conditions such as variations in lighting, angle, and partial occlusion.


---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Step-by-Step Guide for Evaluation](#step-by-step-guide-for-evaluation)
  - [1. Setup and Installation](#1-setup-and-installation)
  - [2. Running Inference (Official Evaluation)](#2-running-inference-official-evaluation)
- [Additional Scripts](#additional-scripts)
  - [Visual Demonstration](#visual-demonstration)
  - [Reproducing the Training Process](#reproducing-the-training-process)
- [Model Performance](#model-performance)

---

## 📖 Project Overview

The primary objective of this project is to solve the detection challenge by identifying all QR codes in a given image and outputting their bounding box coordinates. The solution is built using the `ultralytics` library, leveraging a pre-trained YOLOv8n model which is subsequently fine-tuned on a custom-annotated dataset derived from the competition's training images.

- **Model:** YOLOv8n (nano version)
- **Framework:** PyTorch (via `ultralytics`)

---

## 📂 Repository Structure

The project is organized following the recommended hackathon structure for clarity and reproducibility:

```

├── README.md                \# This instruction file
├── requirements.txt         \# List of all Python libraries needed
├── train.py                 \# The script used to train the model from scratch
├── infer.py                 \# The primary script for running inference for the official submission
├── visual\_test.py           \# An optional script for visual demonstration
│
└── weights/
└── best.pt              \# The final, trained model weights for inference

````

---

## 📝 Step-by-Step Guide for Evaluation

This section provides the exact steps required for an evaluator to set up the environment and run inference on a new set of images.

### 1. Setup and Installation

**1.1. Clone the Repository**
Open your terminal and clone this GitHub repository to your local machine.
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
````

**1.2. Create a Virtual Environment (Recommended)**
It is best practice to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**1.3. Install Dependencies**
All required libraries are listed in `requirements.txt`. Install them using pip.

```bash
pip install -r requirements.txt
```

### 2\. Running Inference (Official Evaluation)

The `infer.py` script is the primary tool for evaluation. It silently processes a folder of images and generates the required `submission.json` file.

**To run inference, execute the following command in your terminal:**
You must provide a path to the folder containing the test images.

```bash
python infer.py --input <path_to_your_test_images> --output submission.json --model_weights weights/best.pt
```

**Example:**
If you have a folder named `demo_images` in your project, the command would be:

```bash
python infer.py --input demo_images/ --output submission.json --model_weights weights/best.pt
```

This command will create the `submission.json` file in your project directory.

-----

## 💡 Additional Scripts

### Visual Demonstration

For a more interactive and visual demonstration of the model's capabilities, you can use the `visual_test.py` script. It processes a single image and displays the output with bounding boxes drawn on it.

**To run the visual test:**

```bash
python visual_test.py --image "demo_images/your_image_name.jpg" --model_weights "weights/best.pt"
```

### Reproducing the Training Process

The `train.py` script is included for reproducibility. It allows retraining the model from scratch.

**Prerequisites:**

  - The original dataset must be unzipped and available.
  - An annotation `.zip` file in YOLO format must be provided.

**To run training:**

```bash
python train.py --annotations <path_to_labels.zip> --images <path_to_original_train_images> --project_path <folder_to_save_results>
```

-----

## 📊 Model Performance

The final model was trained for **300 epochs** on a custom-annotated dataset. It achieved the following high performance on its validation set, demonstrating that it successfully learned the features from the training data.

  - **Precision:** 0.999
  - **Recall:** 1.0
  - **mAP50 (B):** 0.995

*(These values were retrieved from the `results.csv` of the `QR_Detection_Long_Run` training session).*

```
```
