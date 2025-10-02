import os
import shutil
import yaml
from ultralytics import YOLO
from google.colab import drive

def main():
    # --- 1. SETUP ---
    print("Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=True)
    
    # --- Define all paths ---
    zip_filename = "your_labels_file.zip" # Placeholder name
    unzip_dir = '/content/yolo_export'
    yolo_dataset_path = "/content/QR_YOLO_Dataset"
    project_path = "/content/drive/MyDrive/MultiQR_Hackathon_Project"
    source_images_path = "/content/MultiQR_dataset/QR_Dataset/train_images"

    print("This script expects you to upload your annotation zip file to /content/")
    if not os.path.exists(zip_filename):
        print(f"❌ ERROR: Zip file not found at /content/{zip_filename}.")
        print("Please upload your annotation zip file and update the 'zip_filename' variable.")
        return

    # --- 2. UNZIP & ORGANIZE ---
    print("Unzipping and organizing data...")
    if os.path.exists(unzip_dir): shutil.rmtree(unzip_dir)
    os.makedirs(unzip_dir, exist_ok=True)
    os.system(f'unzip -q "{zip_filename}" -d "{unzip_dir}"')

    image_source_path = ""
    label_source_path = ""
    for root, dirs, files in os.walk(unzip_dir):
        if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files): image_source_path = root
        if any(f.endswith('.txt') for f in files): label_source_path = root
            
    if not image_source_path or not label_source_path:
        print("❌ ERROR: Could not find both images and labels in the zip file.")
        return

    if os.path.exists(yolo_dataset_path): shutil.rmtree(yolo_dataset_path)
    final_images_path = os.path.join(yolo_dataset_path, "images/train")
    final_labels_path = os.path.join(yolo_dataset_path, "labels/train")
    os.makedirs(final_images_path, exist_ok=True)
    os.makedirs(final_labels_path, exist_ok=True)

    shutil.copytree(image_source_path, final_images_path, dirs_exist_ok=True)
    shutil.copytree(label_source_path, final_labels_path, dirs_exist_ok=True)
    print("✅ Data preparation complete.")

    # --- 3. TRAIN ---
    data_yaml = {'path': yolo_dataset_path, 'train': 'images/train', 'val': 'images/train',
                 'nc': 1, 'names': ['QR_Code']}
    yaml_file_path = os.path.join(yolo_dataset_path, "data.yaml")
    with open(yaml_file_path, 'w') as f: yaml.dump(data_yaml, f)

    model = YOLO('yolov8n.pt')

    print("\n🚀 Starting model training...")
    model.train(
        data=yaml_file_path,
        epochs=100,
        patience=20,
        imgsz=640,
        project=project_path,
        name="Final_Hackathon_Run"
    )
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()
