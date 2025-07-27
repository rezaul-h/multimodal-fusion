# organize_dataset.py
import os
import csv
from collections import defaultdict

DATA_ROOT = './data/raw'  # Change if your root folder is different
OUTPUT_CSV = './data/metadata.csv'

def find_pairs_for_class(class_folder):
    audio_dir = os.path.join(class_folder, 'audio')
    image_dir = os.path.join(class_folder, 'image')
    if not os.path.isdir(audio_dir) or not os.path.isdir(image_dir):
        print(f"WARNING: Skipping {class_folder} (audio or image folder missing)")
        return []

    audio_files = set(os.path.splitext(f)[0] for f in os.listdir(audio_dir) if not f.startswith('.') and os.path.isfile(os.path.join(audio_dir, f)))
    image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if not f.startswith('.') and os.path.isfile(os.path.join(image_dir, f)))
    common = audio_files & image_files

    pairs = []
    for base in sorted(common):
        audio_path = os.path.abspath(os.path.join(audio_dir, base + '.wav'))
        # Accept multiple image formats (jpg, png, jpeg)
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            candidate = os.path.join(image_dir, base + ext)
            if os.path.isfile(candidate):
                image_path = os.path.abspath(candidate)
                break
        if image_path and os.path.isfile(audio_path):
            pairs.append((os.path.basename(class_folder), audio_path, image_path))
        else:
            print(f"WARNING: Pair {base} in {class_folder} missing a valid image or audio file.")
    # Log unmatched files
    unmatched_audio = audio_files - common
    unmatched_image = image_files - common
    if unmatched_audio:
        print(f"Class '{os.path.basename(class_folder)}': {len(unmatched_audio)} audio(s) have no image pair.")
    if unmatched_image:
        print(f"Class '{os.path.basename(class_folder)}': {len(unmatched_image)} image(s) have no audio pair.")

    return pairs

def main():
    all_pairs = []
    class_folders = [os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT)
                     if os.path.isdir(os.path.join(DATA_ROOT, d))]

    for class_folder in class_folders:
        pairs = find_pairs_for_class(class_folder)
        all_pairs.extend(pairs)

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'audio_path', 'image_path'])
        for row in all_pairs:
            writer.writerow(row)

    print(f"\nWrote metadata for {len(all_pairs)} paired samples to {OUTPUT_CSV}")
    print(f"Number of classes found: {len(class_folders)}")

    # Optional: Print class distribution
    class_count = defaultdict(int)
    for c, _, _ in all_pairs:
        class_count[c] += 1
    print("Sample count per class:")
    for c in sorted(class_count):
        print(f"  {c}: {class_count[c]} pairs")

if __name__ == "__main__":
    main()
