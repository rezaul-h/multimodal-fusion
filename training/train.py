# training/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----- CONFIGURATION -----
FEATURE_DIR = './data/features_balanced/train'
VAL_FEATURE_DIR = './data/features_balanced/val'
CLASS_WEIGHTS_PATH = './data/class_weights.npy'
NUM_CLASSES = 21
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = 'hybrid_xlstm'  # hybrid_xlstm, hybrid_lstm, tensor, film

# --- Import model components ---
from models.vision_xception import MiniXception
from models.audio_lstm import AudioLSTM
from models.audio_xlstm import xLSTMBranch
from models.fusion_hybrid import HybridFusion
from models.fusion_tensor import TensorFusion
from models.fusion_film import FiLMFusion

# ----- DATASET -----
class MultiModalFeatureDataset(Dataset):
    def __init__(self, csv_path, feature_dir, label2idx=None):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = feature_dir
        self.samples = self.df['sample'].tolist()
        self.labels = self.df['class'].tolist()
        self.label2idx = label2idx or {c: i for i, c in enumerate(sorted(set(self.labels)))}
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path = os.path.join(self.feature_dir, self.samples[idx])
        feat = np.load(path).astype(np.float32)
        label = self.label2idx[self.labels[idx]]
        return torch.tensor(feat), label

def get_loaders(train_csv, val_csv, feature_dir, val_feature_dir, batch_size):
    train_ds = MultiModalFeatureDataset(train_csv, feature_dir)
    val_ds = MultiModalFeatureDataset(val_csv, val_feature_dir, train_ds.label2idx)
    class_counts = np.bincount([train_ds.label2idx[c] for c in train_ds.labels])
    class_weights = 1. / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in [train_ds.label2idx[c] for c in train_ds.labels]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_ds.label2idx

# ----- MODEL WRAPPER -----
class MultiModalClassifier(nn.Module):
    def __init__(self, image_dim, audio_dim, fusion='hybrid_xlstm', out_dim=128, n_classes=21):
        super().__init__()
        # You may need to set these dims based on your extracted feature size!
        self.fusion = fusion
        self.img_proj = nn.Linear(image_dim, out_dim)
        if 'xlstm' in fusion:
            self.aud_proj = nn.Linear(audio_dim, out_dim)
        else:
            self.aud_proj = nn.Linear(audio_dim, out_dim)
        if fusion.startswith('hybrid'):
            self.fuse = HybridFusion(out_dim, out_dim, out_dim)
        elif fusion == 'tensor':
            self.fuse = TensorFusion(out_dim, out_dim, out_dim)
        elif fusion == 'film':
            self.fuse = FiLMFusion(out_dim, out_dim, out_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")
        self.classifier = nn.Linear(out_dim, n_classes)
    def forward(self, x):
        # x: (B, D) where D = image+audio feature concat
        half = x.shape[1] // 2
        v = self.img_proj(x[:, :half])
        a = self.aud_proj(x[:, half:])
        fused = self.fuse(v, a)
        return self.classifier(fused)

# ----- TRAINING -----
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for x, y in tqdm(loader, desc="Val "):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n

def main():
    # ---- Loaders and weights ----
    train_csv = './data/splits/train_balanced.csv'
    val_csv = './data/splits/val.csv'
    train_loader, val_loader, label2idx = get_loaders(train_csv, val_csv, FEATURE_DIR, VAL_FEATURE_DIR, BATCH_SIZE)
    class_weights = np.load(CLASS_WEIGHTS_PATH)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    # ---- Infer feature dims ----
    feat_sample, _ = next(iter(train_loader))
    feat_dim = feat_sample.shape[1]
    image_dim = audio_dim = feat_dim // 2  # assuming equal split
    # ---- Model and optimizer ----
    model = MultiModalClassifier(image_dim, audio_dim, fusion=MODEL_TYPE, out_dim=128, n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ---- Training loop ----
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | Val: Loss={val_loss:.4f} Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), './checkpoint_best.pth')
            best_val_acc = val_acc

if __name__ == '__main__':
    main()
