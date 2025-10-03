import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import json

# --- QR Utilities ---
ALPHANUMERIC_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_mask_pattern(mask_id):
    if mask_id == 5:
        return lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0
    else:
        return lambda r, c: False

def unmask_grid(grid, mask_id):
    unmasked = grid.copy()
    mask_func = get_mask_pattern(mask_id)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if mask_func(r, c):
                unmasked[r, c] = 1 - unmasked[r, c]
    return unmasked

def read_data_bits(grid):
    size = grid.shape[0]
    bits = []
    c = size - 1
    upward = True
    while c > 0:
        if c == 6:
            c -= 1
        for r in (range(size-1, -1, -1) if upward else range(size)):
            for dc in [0, -1]:
                if not ((r < 9 and dc == -1 and c <= 8) or
                        (r >= size-8 and c <= 8) or
                        (r < 9 and c >= size-8)):
                    bits.append(grid[r, c+dc])
        c -= 2
        upward = not upward
    return bits

def decode_alphanumeric(bits):
    text = ""
    i = 0
    while i + 10 < len(bits):
        chunk = bits[i:i+11]
        val = int("".join(str(b) for b in chunk), 2)
        if val >= 45*45:
            break
        text += ALPHANUMERIC_CHARSET[val // 45]
        text += ALPHANUMERIC_CHARSET[val % 45]
        i += 11
    if i + 5 < len(bits):
        val = int("".join(str(b) for b in bits[i:i+6]), 2)
        if val < len(ALPHANUMERIC_CHARSET):
            text += ALPHANUMERIC_CHARSET[val]
    return text

def decode_binary_grid(grid):
    mask_id = 5
    unmasked = unmask_grid(grid, mask_id)
    data_bits = read_data_bits(unmasked)
    decoded_text = decode_alphanumeric(data_bits)
    return decoded_text

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Multi-QR Detection & Decoding")
parser.add_argument("--input", type=str, required=True, help="Input image or folder path")
parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
parser.add_argument("--model_weights", type=str, required=True, help="YOLO model weights path")
args = parser.parse_args()

# --- Load YOLO Model ---
if not os.path.exists(args.model_weights):
    print(f" ERROR: Model not found at {args.model_weights}.")
    exit()
model = YOLO(args.model_weights)
print("YOLO model loaded successfully.")

# --- Prepare Input Images ---
input_paths = []
if os.path.isdir(args.input):
    for f in os.listdir(args.input):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            input_paths.append(os.path.join(args.input, f))
else:
    input_paths = [args.input]

results_list = []

# --- Process Each Image ---
for img_path in input_paths:
    print(f"\nProcessing: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Failed to load {img_path}")
        continue

    yolo_results = model.predict(source=image, verbose=False, conf=0.25)
    qr_count = 0

    for result in yolo_results:
        if len(result.boxes) == 0:
            print("No QR codes detected.")
            continue

        for box in result.boxes:
            qr_count += 1
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            qr_crop = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

            gray = cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            corners = cv2.approxPolyDP(largest, 0.04 * peri, True)
            if len(corners) != 4:
                continue

            ordered_corners = order_points(corners.reshape(4,2))
            dest_pts = np.array([[0,0],[210-1,0],[210-1,210-1],[0,210-1]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(ordered_corners, dest_pts)
            warped = cv2.warpPerspective(gray, matrix, (210,210))
            _, warped_thresh = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

            grid_size = 21
            module_size = warped_thresh.shape[0] // grid_size
            qr_grid = np.zeros((grid_size, grid_size), dtype=int)
            for y in range(grid_size):
                for x in range(grid_size):
                    cx = int((x*module_size)+(module_size/2))
                    cy = int((y*module_size)+(module_size/2))
                    qr_grid[y, x] = 1 if warped_thresh[cy, cx] < 128 else 0

            decoded_text = decode_binary_grid(qr_grid)
            results_list.append({
                "image": os.path.basename(img_path),
                "qr_index": qr_count,
                "decoded_text": decoded_text
            })
            print(f"QR #{qr_count}: {decoded_text}")

# --- Save JSON Results ---
with open(args.output, "w") as f:
    json.dump(results_list, f, indent=4)
print(f"\n All results saved to {args.output}")
