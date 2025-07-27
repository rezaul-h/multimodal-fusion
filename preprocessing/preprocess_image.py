# preprocessing/preprocess_image.py

import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from PIL import Image
import csv
from tqdm import tqdm

# SETTINGS
INPUT_SPLIT = './data/splits/train.csv'  # Change to val/test for other splits
OUTPUT_ROOT = './data/preprocessed_images/train'
TARGET_SIZE = (128, 128)  # Resize all images
TV_WEIGHT = 0.2  # Strength of TV denoising, adjust as needed

def preprocess_image(img_path):
    # Load as RGB
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # DRTV (using TV denoising as a stand-in for DRTV)
    img = denoise_tv_chambolle(img, weight=TV_WEIGHT, multichannel=True)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Normalization (0-1 float)
    img = img.astype(np.float32) / 255.0

    return img

def save_preprocessed_image(img, save_path):
    # img: numpy array in [0,1]
    img_uint8 = (img * 255).astype(np.uint8)
    im = Image.fromarray(img_uint8)
    im.save(save_path)

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    # Read input split
    with open(INPUT_SPLIT, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    num_processed = 0
    num_skipped = 0
    for row in tqdm(rows, desc="Processing images"):
        class_name = row['class']
        img_path = row['image_path']
        # Set up output folder
        out_dir = os.path.join(OUTPUT_ROOT, class_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Output file name
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, base + ".png")
        try:
            img = preprocess_image(img_path)
            save_preprocessed_image(img, out_path)
            num_processed += 1
        except Exception as e:
            print(f"Failed for {img_path}: {e}")
            num_skipped += 1

    print(f"Done! Processed {num_processed} images, skipped {num_skipped}.")

if __name__ == '__main__':
    main()
