import os
import cv2
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    GaussNoise, CLAHE, RandomResizedCrop, ShiftScaleRotate
)
from audiomentations import Compose as AudioCompose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# -------- IMAGE AUGMENTATION -------- #
image_aug = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    RandomBrightnessContrast(p=0.5),
    Rotate(limit=30, p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    CLAHE(p=0.2),
    GaussNoise(p=0.2)
])

def augment_image(img_path, save_dir, num_aug=3, size=(128,128)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base = os.path.splitext(os.path.basename(img_path))[0]
    for i in range(num_aug):
        aug = image_aug(image=img)['image']
        aug = cv2.resize(aug, size)
        out_path = os.path.join(save_dir, f"{base}_aug{i+1}.png")
        cv2.imwrite(out_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

# -------- AUDIO AUGMENTATION -------- #
audio_aug = AudioCompose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.4),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
    Shift(min_fraction=-0.1, max_fraction=0.1, p=0.3),
])

def augment_audio(audio_path, save_dir, num_aug=3, sr=16000, duration=3.0):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    if len(y) < int(sr*duration):
        y = np.pad(y, (0, int(sr*duration)-len(y)), mode='constant')
    else:
        y = y[:int(sr*duration)]
    for i in range(num_aug):
        y_aug = audio_aug(samples=y, sample_rate=sr)
        out_path = os.path.join(save_dir, f"{base}_aug{i+1}.wav")
        sf.write(out_path, y_aug, sr)

# -------- MAIN PIPELINE -------- #
def process_dataset(dataset_root, num_aug=3, img_size=(128,128), sr=16000, duration=3.0):
    classes = sorted(os.listdir(dataset_root))
    for cls in tqdm(classes, desc='Classes'):
        class_dir = os.path.join(dataset_root, cls)
        img_dir = os.path.join(class_dir, 'image')
        aud_dir = os.path.join(class_dir, 'audio')
        # Augment images
        if os.path.isdir(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            save_img_dir = os.path.join(img_dir, 'augmented')
            os.makedirs(save_img_dir, exist_ok=True)
            for img in tqdm(img_files, desc=f'  {cls} images'):
                augment_image(os.path.join(img_dir, img), save_img_dir, num_aug=num_aug, size=img_size)
        # Augment audio
        if os.path.isdir(aud_dir):
            aud_files = [f for f in os.listdir(aud_dir) if f.lower().endswith('.wav')]
            save_aud_dir = os.path.join(aud_dir, 'augmented')
            os.makedirs(save_aud_dir, exist_ok=True)
            for af in tqdm(aud_files, desc=f'  {cls} audio'):
                augment_audio(os.path.join(aud_dir, af), save_aud_dir, num_aug=num_aug, sr=sr, duration=duration)

if __name__ == "__main__":
    DATASET_ROOT = './data/dataset'
    process_dataset(DATASET_ROOT, num_aug=3, img_size=(128,128), sr=16000, duration=3.0)
    print("Augmentation complete.")
