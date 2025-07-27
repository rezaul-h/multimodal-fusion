# utils/smote.py

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# SETTINGS
SPLIT_CSV = './data/splits/train.csv'
AUDIO_FEATURE_DIR = './data/features_audio/train'
IMAGE_FEATURE_DIR = './data/features_image/train'
OUTPUT_FEATURE_DIR = './data/features_balanced/train'
OUTPUT_CSV = './data/splits/train_balanced.csv'
USE_COMBINED = True   # If True, concatenate audio and image features for SMOTE

def load_feature(row, use_combined=True):
    class_name = row['class']
    base = os.path.splitext(os.path.basename(row['audio_path']))[0]
    # Audio
    audio_path = os.path.join(AUDIO_FEATURE_DIR, class_name, f"{base}_audio.npy")
    audio_feat = np.load(audio_path)
    # Image
    base_img = os.path.splitext(os.path.basename(row['image_path']))[0]
    image_path = os.path.join(IMAGE_FEATURE_DIR, class_name, f"{base_img}_sift_lbp.npy")
    image_feat = np.load(image_path)
    # Combine or select
    if use_combined:
        return np.concatenate([image_feat, audio_feat])
    else:
        return image_feat  # or audio_feat

def main():
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    df = pd.read_csv(SPLIT_CSV)
    features, labels, orig_rows = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Loading features'):
        try:
            feat = load_feature(row, use_combined=USE_COMBINED)
            features.append(feat)
            labels.append(row['class'])
            orig_rows.append(row)
        except Exception as e:
            print(f"Skip {row['audio_path']} / {row['image_path']}: {e}")

    features = np.stack(features)
    print(f"Loaded {features.shape[0]} samples, feature dim {features.shape[1]}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(features, labels)
    print(f"After SMOTE: {X_res.shape[0]} samples, {len(set(y_res))} classes")

    # Save new features and updated CSV
    out_rows = []
    for i, (vec, label) in enumerate(zip(X_res, y_res)):
        out_name = f"sample_{i:06d}.npy"
        np.save(os.path.join(OUTPUT_FEATURE_DIR, out_name), vec)
        out_rows.append({'sample': out_name, 'class': label})

    pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Balanced features written to {OUTPUT_FEATURE_DIR}, CSV: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
