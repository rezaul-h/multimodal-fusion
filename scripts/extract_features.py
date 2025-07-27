import os
import numpy as np
import torch
from tqdm import tqdm

# --- Import your preprocessors and feature extractors ---
from preprocessing.preprocess_image import preprocess_image
from preprocessing.preprocess_audio import preprocess_audio
from features.extract_audio import extract_audio_features
from features.extarct_ridgelet import extract_ridgelet_features
from models.vision_xception import XceptionBackbone

import librosa

def get_image_files(image_dir):
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_audio_files(audio_dir):
    return [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]

def extract_and_save_features(
        dataset_root, 
        save_dir, 
        image_model_ckpt=None, 
        img_size=(128, 128), 
        sr=16000, 
        audio_duration=3.0,
        use_ridgelet=False
    ):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = XceptionBackbone(pretrained=True).to(device)
    if image_model_ckpt:
        model.load_state_dict(torch.load(image_model_ckpt, map_location=device))
    model.eval()

    classes = sorted(os.listdir(dataset_root))
    for cls_idx, cls in enumerate(tqdm(classes, desc="Classes")):
        class_dir = os.path.join(dataset_root, cls)
        img_dir = os.path.join(class_dir, 'image')
        aud_dir = os.path.join(class_dir, 'audio')
        if not (os.path.isdir(img_dir) and os.path.isdir(aud_dir)):
            continue

        img_files = get_image_files(img_dir)
        aud_files = get_audio_files(aud_dir)

        for base_file in tqdm(sorted(set([os.path.splitext(os.path.basename(f))[0] for f in img_files]).intersection(
                                    set([os.path.splitext(os.path.basename(f))[0] for f in aud_files])),
                              desc=f'  {cls} pairs'):
            img_path = os.path.join(img_dir, base_file + ".png")
            if not os.path.exists(img_path): img_path = os.path.join(img_dir, base_file + ".jpg")
            aud_path = os.path.join(aud_dir, base_file + ".wav")

            # --- Vision features ---
            img = preprocess_image(img_path, img_size=img_size)  # Should output tensor (C,H,W), normalized
            img_tensor = img.unsqueeze(0).to(device)
            with torch.no_grad():
                vis_feat = model(img_tensor)   # (1, feat_dim)
            vis_feat = vis_feat.cpu().numpy().squeeze()   # (feat_dim,)

            # --- Audio features ---
            audio = preprocess_audio(aud_path, sr=sr, duration=audio_duration)  # Should output 1D np.array
            aud_feat = extract_audio_features(audio, sr=sr)                     # e.g. MFCC, PSSF, etc.

            # --- Ridgelet (optional) ---
            ridge_feat = None
            if use_ridgelet:
                ridge_feat = extract_ridgelet_features(audio, sr=sr)

            # --- Save all features ---
            np.savez(
                os.path.join(save_dir, f"{cls}_{base_file}.npz"),
                vision=vis_feat,
                audio=aud_feat,
                ridgelet=ridge_feat if use_ridgelet else np.zeros(1),
                label=cls_idx
            )

if __name__ == "__main__":
    DATASET_ROOT = './data/dataset'
    SAVE_DIR = './data/features'
    extract_and_save_features(DATASET_ROOT, SAVE_DIR, img_size=(128,128), use_ridgelet=True)
    print("Feature extraction complete.")
