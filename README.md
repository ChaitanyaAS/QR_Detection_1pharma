%%writefile README.md
# Multi-QR Code Detection for Medicine Packs

This project is a submission for the **Multi-QR Code Recognition for Medicine Packs Hackathon**. It uses a fine-tuned YOLOv8 model to accurately detect the location of multiple QR codes on pharmaceutical packaging.

---

## Repository Structure

The project is organized as follows:

your-repo-name/
│
├── README.md
├── requirements.txt
├── train.py
├── infer.py
│
└── weights/
└── best.pt


---

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Training

The `train.py` script fine-tunes a YOLOv8 model on a custom dataset.

**Prerequisites:**
1.  Unzip the main dataset (`Multi-QR-Code-Detection.zip`) so the `train_images` are available in your environment.
2.  Provide an annotation `.zip` file (in YOLO format, containing only the `.txt` labels).

**To run training:**
```bash
python train.py --annotations <path_to_labels.zip> --project_path <path_to_save_run> --run_name "My_Training_Run"
Inference
The infer.py script generates the final submission.json file.

To run inference, use the following command:

Bash

python infer.py --input <path_to_test_images> --output submission.json --model_weights weights/best.pt
Model Performance
The model was trained for 300 epochs on a custom-annotated dataset. The final model (QR_Detection_Long_Run) achieved the following performance on its validation set:

Precision: 0.999

Recall: 1.0

mAP50 (B): 0.995
