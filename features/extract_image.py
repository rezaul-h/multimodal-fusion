# features/extract_image.py

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

PREPROCESSED_IMG_ROOT = './data/preprocessed_images/train'   # Change for val/test
FEATURES_OUT_ROOT = './data/features_image/train'
LBP_RADIUS = 2
LBP_N_POINTS = 8 * LBP_RADIUS

def extract_sift(img_gray):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    # If too few keypoints, pad with zeros
    if descriptors is None or len(descriptors) == 0:
        descriptors = np.zeros((1, 128))
    # Mean-pool to get fixed-length feature
    sift_feature = np.mean(descriptors, axis=0)
    return sift_feature

def extract_lbp(img_gray):
    lbp = local_binary_pattern(img_gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    # Histogram
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return lbp_hist

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load: {img_path}")
    img = cv2.resize(img, (128, 128))
    sift_feat = extract_sift(img)
    lbp_feat = extract_lbp(img)
    sift_lbp_feat = np.concatenate([sift_feat, lbp_feat])
    return {'sift': sift_feat, 'lbp': lbp_feat, 'sift_lbp': sift_lbp_feat}

def save_features(feat_dict, out_dir, base):
    for name, arr in feat_dict.items():
        out_path = os.path.join(out_dir, f"{base}_{name}.npy")
        np.save(out_path, arr)

def main():
    for class_name in os.listdir(PREPROCESSED_IMG_ROOT):
        class_dir = os.path.join(PREPROCESSED_IMG_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
        out_dir = os.path.join(FEATURES_OUT_ROOT, class_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        img_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        for img_file in tqdm(img_files, desc=f"Processing {class_name}"):
            img_path = os.path.join(class_dir, img_file)
            base = os.path.splitext(img_file)[0]
            try:
                feats = process_image(img_path)
                save_features(feats, out_dir, base)
            except Exception as e:
                print(f"Failed for {img_path}: {e}")

if __name__ == '__main__':
    main()
