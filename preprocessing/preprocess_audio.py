# preprocessing/preprocess_audio.py

import os
import librosa
import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm
import csv

# SETTINGS
INPUT_SPLIT = './data/splits/train.csv'  # Change to val/test as needed
AUDIO_OUT_ROOT = './data/preprocessed_audio/train'
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
N_MFCC = 20

def denoise_audio(audio):
    # Simple median filter denoising
    return medfilt(audio, kernel_size=5)

def pad_truncate(audio, target_length):
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        pad_width = target_length - len(audio)
        return np.pad(audio, (0, pad_width), mode='constant')
    else:
        return audio

def extract_features(audio, sr):
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    # Pitch-Synchronous Speech Features (simplified: zero crossing + RMS)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)
    # Ridgelet placeholder: Use a flat array for now (replace with actual extraction if available)
    ridgelet = np.mean(mfcc, axis=1, keepdims=True)
    # Stack all features (optional: concatenate, or save separately)
    features = {
        "mfcc": mfcc,
        "spectral_centroid": spec_centroid,
        "zcr": zcr,
        "rms": rms,
        "ridgelet": ridgelet
    }
    return features

def save_features(features, out_dir, base_name):
    for feat_name, arr in features.items():
        out_path = os.path.join(out_dir, f"{base_name}_{feat_name}.npy")
        np.save(out_path, arr)

def main():
    if not os.path.exists(AUDIO_OUT_ROOT):
        os.makedirs(AUDIO_OUT_ROOT)
    # Read input split
    with open(INPUT_SPLIT, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    num_processed = 0
    num_skipped = 0
    for row in tqdm(rows, desc="Processing audio"):
        class_name = row['class']
        audio_path = row['audio_path']
        out_dir = os.path.join(AUDIO_OUT_ROOT, class_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            # Load
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            audio = denoise_audio(audio)
            target_len = int(SAMPLE_RATE * DURATION)
            audio = pad_truncate(audio, target_len)
            # Features
            features = extract_features(audio, SAMPLE_RATE)
            save_features(features, out_dir, base)
            num_processed += 1
        except Exception as e:
            print(f"Failed for {audio_path}: {e}")
            num_skipped += 1
    print(f"Done! Processed {num_processed} audio files, skipped {num_skipped}.")

if __name__ == "__main__":
    main()
