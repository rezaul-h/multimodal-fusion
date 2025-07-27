import os
from tqdm import tqdm
import numpy as np
import soundfile as sf
import cv2

# Import your own preprocessors
from preprocessing.preprocess_image import preprocess_image
from preprocessing.preprocess_audio import preprocess_audio

def get_image_files(image_dir):
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_audio_files(audio_dir):
    return [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]

def preprocess_dataset(
        dataset_root,
        save_root,
        img_size=(128, 128),
        sr=16000,
        duration=3.0
    ):
    os.makedirs(save_root, exist_ok=True)
    classes = sorted(os.listdir(dataset_root))
    for cls in tqdm(classes, desc="Classes"):
        class_dir = os.path.join(dataset_root, cls)
        img_dir = os.path.join(class_dir, 'image')
        aud_dir = os.path.join(class_dir, 'audio')
        if not (os.path.isdir(img_dir) and os.path.isdir(aud_dir)):
            continue

        # Prepare output folders
        out_img_dir = os.path.join(save_root, cls, 'image')
        out_aud_dir = os.path.join(save_root, cls, 'audio')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_aud_dir, exist_ok=True)

        # Process images
        for img_path in tqdm(get_image_files(img_dir), desc=f"  {cls} images"):
            img = preprocess_image(img_path, img_size=img_size)  # Should return np.array or tensor
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_img_path = os.path.join(out_img_dir, f"{base}.png")
            # Convert tensor to numpy if needed
            if hasattr(img, 'cpu') and hasattr(img, 'numpy'):
                img = img.cpu().numpy()
                if img.shape[0] in [1,3]:  # C,H,W --> H,W,C
                    img = np.transpose(img, (1,2,0))
            img = np.uint8(np.clip(img*255, 0, 255)) if img.max() <= 1.0 else np.uint8(img)
            cv2.imwrite(out_img_path, img)

        # Process audio
        for aud_path in tqdm(get_audio_files(aud_dir), desc=f"  {cls} audio"):
            audio = preprocess_audio(aud_path, sr=sr, duration=duration)  # Should return np.array (float32)
            base = os.path.splitext(os.path.basename(aud_path))[0]
            out_aud_path = os.path.join(out_aud_dir, f"{base}.wav")
            sf.write(out_aud_path, audio, sr)

if __name__ == "__main__":
    DATASET_ROOT = './data/dataset'           # Path to original dataset
    SAVE_ROOT = './data/preprocessed'         # Path to preprocessed output
    preprocess_dataset(DATASET_ROOT, SAVE_ROOT, img_size=(128,128), sr=16000, duration=3.0)
    print("Preprocessing complete.")
