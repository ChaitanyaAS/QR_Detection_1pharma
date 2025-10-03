import cv2
import numpy as np
from ultralytics import YOLO
import os
import requests
from PIL import Image
import argparse

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def visualize(image_path_or_url, model_path):
    model = YOLO(model_path)
    print("Trained model loaded successfully.")

    original_image = None
    if image_path_or_url.startswith('http'):
        try:
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            pil_image = Image.open(response.raw).convert('RGB')
            original_image = np.array(pil_image)[:, :, ::-1].copy()
            print(" Successfully downloaded image.")
        except Exception as e:
            print(f" ERROR: Failed to download image: {e}")
            return
    else:
        if not os.path.exists(image_path_or_url):
            print(f"❌ ERROR: Image not found at {image_path_or_url}")
            return
        original_image = cv2.imread(image_path_or_url)
        print("Successfully loaded local image.")

    if original_image is not None:
        results = model.predict(source=original_image, verbose=False, conf=0.25)
        
        image_with_all_boxes = original_image.copy()
        qr_count = 0
        
        for result in results:
            if len(result.boxes) == 0:
                print("\nNo QR codes were detected in this image.")
                break

            print(f"\nFound {len(result.boxes)} total QR codes.")
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(image_with_all_boxes, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            
            print("\n--- Full Image with All Detections ---")
            cv2.imshow("Detections", image_with_all_boxes)
            cv2.waitKey(0) # Wait for a key press to close the window
            cv2.destroyAllWindows()

            for box in result.boxes:
                qr_count += 1
                print(f"\n--- Processing Bonus Challenge for QR Code #{qr_count} ---")
                
                # ... (rest of the bonus challenge code for binary extraction) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual QR Code Decoder")
    parser.add_argument('--image', type=str, required=True, help="Path or URL to the image to test.")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to the trained model weights (best.pt).")
    args = parser.parse_args()
    visualize(args.image, args.model_weights)
