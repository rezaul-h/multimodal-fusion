# features/extract_ridgelet.py

import os
import numpy as np
import librosa
import soundfile as sf
import pywt
from scipy.ndimage import rotate
from skimage.transform import radon
from tqdm import tqdm

def load_audio_mono(filepath, sr=16000, duration=3.0):
    # Loads audio, mono, fixed length
    x, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration)
    if len(x) < int(sr * duration):
        x = np.pad(x, (0, int(sr * duration) - len(x)), mode='constant')
    else:
        x = x[:int(sr * duration)]
    return x

def audio_to_spectrogram(audio, sr=16000, n_fft=1024, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def ridgelet_transform(spectrogram, theta_list=None, wavelet='db4', level=3):
    # Apply Radon transform, then 1D DWT along each projection (Ridgelet)
    if theta_list is None:
        theta_list = np.linspace(0., 180., max(spectrogram.shape), endpoint=False)
    radon_image = radon(spectrogram, theta=theta_list, circle=False)
    ridgelet_coeffs = []
    for i in range(radon_image.shape[1]):
        # 1D DWT for each projection
        c = pywt.wavedec(radon_image[:, i], wavelet, level=level)
        coeffs_flat = np.concatenate([ci.ravel() for ci in c])
        ridgelet_coeffs.append(coeffs_flat)
    ridgelet_feat = np.stack(ridgelet_coeffs, axis=1)  # (coef_dim, num_angles)
    # Pool to fixed size (e.g., mean and std)
    feat_mean = np.mean(ridgelet_feat, axis=1)
    feat_std = np.std(ridgelet_feat, axis=1)
    return np.concatenate([feat_mean, feat_std], axis=0)

def extract_ridgelet_features_from_audio(audio_path, sr=16000):
    audio = load_audio_mono(audio_path, sr=sr)
    spec = audio_to_spectrogram(audio, sr=sr)
    features = ridgelet_transform(spec)
    return features

def process_dataset(dataset_root, save_dir, sr=16000):
    os.makedirs(save_dir, exist_ok=True)
    classes = sorted(os.listdir(dataset_root))
    for cls in tqdm(classes, desc='Classes'):
        class_path = os.path.join(dataset_root, cls, "audio")
        if not os.path.isdir(class_path): continue
        out_dir = os.path.join(save_dir, cls)
        os.makedirs(out_dir, exist_ok=True)
        audio_files = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
        for af in tqdm(audio_files, desc=f'  {cls} audio'):
            audio_path = os.path.join(class_path, af)
            features = extract_ridgelet_features_from_audio(audio_path, sr=sr)
            base = os.path.splitext(af)[0]
            np.save(os.path.join(out_dir, base + "_ridgelet.npy"), features)

if __name__ == '__main__':
    # Example usage:
    DATASET_ROOT = './data/dataset'
    SAVE_DIR = './data/features_ridgelet'
    process_dataset(DATASET_ROOT, SAVE_DIR)
    print("Ridgelet feature extraction complete.")
