%%writefile train.py
import os
import shutil
import yaml
from ultralytics import YOLO
import argparse

def train_model(annotation_zip, source_images_path, project_path, run_name, epochs, batch_size, patience, imgsz):
    """
    Prepares the dataset from a zip file and trains a YOLOv8 model.
    """
    # Define paths for temporary data
    unzip_dir = 'yolo_export'
    yolo_dataset_path = "QR_YOLO_Dataset"

    print("--- Data Preparation ---")
    if not os.path.exists(annotation_zip):
        print(f"ERROR: Annotation zip file not found at {annotation_zip}.")
        return

    # Unzip and Organize Data
    print("Unzipping and organizing data...")
    if os.path.exists(unzip_dir): shutil.rmtree(unzip_dir)
    os.makedirs(unzip_dir, exist_ok=True)
    os.system(f'unzip -q "{annotation_zip}" -d "{unzip_dir}"')

    # Find the folder containing the .txt label files
    label_source_path = ""
    for root, dirs, files in os.walk(unzip_dir):
        if any(f.endswith('.txt') for f in files):
            label_source_path = root
            break
            
    if not label_source_path:
        print("ERROR: No .txt label files found inside the zip file.")
        return

    # Recreate the final dataset folder
    if os.path.exists(yolo_dataset_path): shutil.rmtree(yolo_dataset_path)
    final_images_path = os.path.join(yolo_dataset_path, "images/train")
    final_labels_path = os.path.join(yolo_dataset_path, "labels/train")
    os.makedirs(final_images_path, exist_ok=True)
    os.makedirs(final_labels_path, exist_ok=True)

    # Match labels with original images and copy them
    copied_images = 0
    for label_file in os.listdir(label_source_path):
        if label_file.endswith('.txt'):
            base_filename = os.path.splitext(label_file)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                image_filename = base_filename + ext
                source_image_file = os.path.join(source_images_path, image_filename)
                if os.path.exists(source_image_file):
                    shutil.copy(source_image_file, final_images_path)
                    shutil.copy(os.path.join(label_source_path, label_file), final_labels_path)
                    copied_images += 1
                    break
    
    print(f"Prepared a dataset with {copied_images} images.")
    if copied_images == 0:
        print("CRITICAL ERROR: No matching images were found for the provided labels.")
        return

    # --- Model Training ---
    print("\n--- Model Training ---")
    data_yaml = {'path': yolo_dataset_path, 'train': 'images/train', 'val': 'images/train',
                 'nc': 1, 'names': ['QR_Code']}
    yaml_file_path = os.path.join(yolo_dataset_path, "data.yaml")
    with open(yaml_file_path, 'w') as f:
        yaml.dump(data_yaml, f)

    model = YOLO('yolov8n.pt')
    
    print("Starting model training...")
    model.train(
        data=yaml_file_path,
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch_size,
        augment=False,
        project=project_path,
        name=run_name
    )
    print("\nTraining complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR Code Detection Training Script")
    parser.add_argument('--annotations', type=str, required=True, help="Path to the annotation zip file (containing .txt labels).")
    parser.add_argument('--images', type=str, required=True, help="Path to the original training images folder.")
    parser.add_argument('--project_path', type=str, required=True, help="Path to save the training run folder (e.g., your Drive folder).")
    parser.add_argument('--run_name', type=str, default="QR_Detection_Run", help="Name for the training run folder.")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--patience', type=int, default=50, help="Patience for early stopping.")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training.")
    
    args = parser.parse_args()
    
    train_model(args.annotations, args.images, args.project_path, args.run_name, args.epochs, args.batch_size, args.patience, args.imgsz)
