# create_splits.py
import os
import csv
import random
from collections import defaultdict

METADATA_CSV = './data/metadata.csv'
OUTPUT_DIR = './data/splits'
TRAIN_RATIO = 0.80
VAL_RATIO = 0.05
TEST_RATIO = 0.15
SEED = 42

def stratified_split(pairs, train_ratio, val_ratio, test_ratio, seed=42):
    # pairs: list of (class, audio_path, image_path)
    random.seed(seed)
    by_class = defaultdict(list)
    for row in pairs:
        by_class[row[0]].append(row)
    train, val, test = [], [], []
    for cls, rows in by_class.items():
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train.extend(rows[:n_train])
        val.extend(rows[n_train:n_train + n_val])
        test.extend(rows[n_train + n_val:])
    return train, val, test

def write_split(split, split_name):
    split_csv = os.path.join(OUTPUT_DIR, f'{split_name}.csv')
    with open(split_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'audio_path', 'image_path'])
        for row in split:
            writer.writerow(row)
    print(f"Wrote {len(split)} samples to {split_csv}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pairs = []
    with open(METADATA_CSV, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            pairs.append(row)

    train, val, test = stratified_split(
        pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)

    write_split(train, 'train')
    write_split(val, 'val')
    write_split(test, 'test')

    print(f"\nSplit sizes — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    main()
