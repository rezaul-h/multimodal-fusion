# utils/class_weights.py

import pandas as pd
import numpy as np
import json
from collections import Counter

# SETTINGS
SPLIT_CSV = './data/splits/train.csv'  # Or train_balanced.csv if using SMOTE
OUTPUT_JSON = './data/class_weights.json'
OUTPUT_NPY = './data/class_weights.npy'

def compute_class_weights(labels):
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)
    # Inverse frequency, normalized so mean weight = 1
    class_weights = {}
    for label in label_counts:
        class_weights[label] = total / (num_classes * label_counts[label])
    return class_weights

def main():
    df = pd.read_csv(SPLIT_CSV)
    # Works with columns 'class' or 'label'
    if 'class' in df.columns:
        labels = df['class'].tolist()
    elif 'label' in df.columns:
        labels = df['label'].tolist()
    else:
        raise ValueError('CSV must contain a "class" or "label" column.')
    class_weights = compute_class_weights(labels)
    print("Class weights:", class_weights)
    # Save as JSON for readability
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(class_weights, f, indent=2)
    # Optionally, save as numpy array (ordered by sorted class names)
    classes_sorted = sorted(class_weights.keys())
    weights_arr = np.array([class_weights[c] for c in classes_sorted], dtype=np.float32)
    np.save(OUTPUT_NPY, weights_arr)
    print(f"Saved: {OUTPUT_JSON}, {OUTPUT_NPY}")
    print(f"Order of classes in NPY: {classes_sorted}")

if __name__ == '__main__':
    main()
