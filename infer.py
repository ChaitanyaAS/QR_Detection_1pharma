import os
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import torch

def run_inference(input_dir, output_path, model_weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = YOLO(model_weights)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model. {e}")
        return

    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"ERROR: No images found in '{input_dir}'")
            return
        print(f"Found {len(image_files)} images to process.")
    except FileNotFoundError:
        print(f"ERROR: Input directory not found at '{input_dir}'")
        return

    submission_data = []
    for image_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_name)
        
        try:
            results = model.predict(source=image_path, device=device, verbose=False, conf=0.25)
        except Exception as e:
            print(f"ERROR: Failed to process image {image_name}. {e}")
            continue

        image_entry = {"image_id": image_name, "qrs": []}
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                bbox_formatted = [int(coord) for coord in xyxy]
                image_entry["qrs"].append({"bbox": bbox_formatted})
        submission_data.append(image_entry)

    try:
        with open(output_path, 'w') as f:
            json.dump(submission_data, f, indent=2)
        print(f"\nSuccess! Submission file saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save submission file. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR Code Detection Inference Script")
    parser.add_argument('--input', type=str, required=True, help="Path to the input directory of images.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output submission.json file.")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to the trained model weights file (best.pt).")
    
    args = parser.parse_args()
    
    run_inference(args.input, args.output, args.model_weights)
