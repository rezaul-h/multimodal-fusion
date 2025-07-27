# evaluation/evaluate.py

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
FEATURE_DIR = './data/features_balanced/test'
CSV_PATH = './data/splits/test.csv'
CHECKPOINT_PATH = './checkpoint_best.pth'
RESULTS_CSV = './evaluation/predictions.csv'
MODEL_TYPE = 'hybrid_xlstm'
NUM_CLASSES = 21
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Model imports ---
from models.vision_xception import MiniXception
from models.audio_lstm import AudioLSTM
from models.audio_xlstm import xLSTMBranch
from models.fusion_hybrid import HybridFusion
from models.fusion_tensor import TensorFusion
from models.fusion_film import FiLMFusion

# --- Dataset ---
class MultiModalFeatureDataset(torch.utils.data.Dataset):
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

# --- Model Wrapper ---
class MultiModalClassifier(nn.Module):
    def __init__(self, image_dim, audio_dim, fusion='hybrid_xlstm', out_dim=128, n_classes=21):
        super().__init__()
        self.fusion = fusion
        self.img_proj = nn.Linear(image_dim, out_dim)
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
        half = x.shape[1] // 2
        v = self.img_proj(x[:, :half])
        a = self.aud_proj(x[:, half:])
        fused = self.fuse(v, a)
        return self.classifier(fused)

def evaluate(model, loader, device, num_classes):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    y_true = labels.numpy()
    # Metrics
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    mcc = matthews_corrcoef(y_true, preds)
    try:
        pr_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
    except Exception:
        pr_auc = float('nan')
    cm = confusion_matrix(y_true, preds)
    return acc, f1, pr_auc, mcc, preds, probs, y_true, cm

def plot_confusion_matrix(cm, class_names, out_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()

def main():
    # --- Load test set ---
    ds = MultiModalFeatureDataset(CSV_PATH, FEATURE_DIR)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    label2idx = ds.label2idx
    idx2label = {v: k for k, v in label2idx.items()}
    # --- Model and checkpoint ---
    feat_sample, _ = next(iter(loader))
    feat_dim = feat_sample.shape[1]
    image_dim = audio_dim = feat_dim // 2
    model = MultiModalClassifier(image_dim, audio_dim, fusion=MODEL_TYPE, out_dim=128, n_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    # --- Evaluation ---
    acc, f1, pr_auc, mcc, preds, probs, y_true, cm = evaluate(model, loader, DEVICE, NUM_CLASSES)
    print(f"Test Acc: {acc:.4f} | F1: {f1:.4f} | PR AUC: {pr_auc:.4f} | MCC: {mcc:.4f}")
    # --- Save results ---
    pred_labels = [idx2label[p] for p in preds]
    true_labels = [idx2label[t] for t in y_true]
    df = pd.DataFrame({'sample': ds.samples, 'true': true_labels, 'pred': pred_labels})
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Predictions saved to {RESULTS_CSV}")
    # --- Confusion Matrix ---
    plot_confusion_matrix(cm, [idx2label[i] for i in range(NUM_CLASSES)], out_path='./evaluation/confusion_matrix.png')

if __name__ == '__main__':
    main()
