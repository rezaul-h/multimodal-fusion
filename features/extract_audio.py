# features/extract_audio.py

import os
import numpy as np
from tqdm import tqdm

PREPROCESSED_AUDIO_ROOT = './data/preprocessed_audio/train'  # Change for val/test
FEATURES_AUDIO_ROOT = './data/features_audio/train'
FEATURE_SET = ['mfcc', 'spectral_centroid', 'zcr', 'rms', 'ridgelet']  # Adjust as needed

def collect_features_for_file(base_path):
    feats = []
    for feat_name in FEATURE_SET:
        npy_path = f"{base_path}_{feat_name}.npy"
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"Missing: {npy_path}")
        arr = np.load(npy_path)
        # Flatten all except MFCC (optionally flatten MFCC if desired)
        arr = arr.flatten()
        feats.append(arr)
    # Concatenate all features
    final_feat = np.concatenate(feats)
    return final_feat

def main():
    for class_name in os.listdir(PREPROCESSED_AUDIO_ROOT):
        class_dir = os.path.join(PREPROCESSED_AUDIO_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
        out_dir = os.path.join(FEATURES_AUDIO_ROOT, class_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        files = [f for f in os.listdir(class_dir) if f.endswith('_mfcc.npy')]
        for mfcc_file in tqdm(files, desc=f"Processing {class_name}"):
            base = mfcc_file[:-9]  # Remove '_mfcc.npy'
            base_path = os.path.join(class_dir, base)
            try:
                feat = collect_features_for_file(base_path)
                np.save(os.path.join(out_dir, f"{base}_audio.npy"), feat)
            except Exception as e:
                print(f"Failed for {base_path}: {e}")

if __name__ == '__main__':
    main()
