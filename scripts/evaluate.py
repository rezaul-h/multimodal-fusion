import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm

# --- Example: import your model and data pipeline ---
from models.vision_xception import XceptionBackbone
from models.audio_xlstm import xLSTMBranch
from fusion.hybrid import HybridFusion
from fusion.film import FiLMFusion
from fusion.tensor import TensorFusion

def load_test_dataset(batch_size=32, num_samples=256, vision_dim=256, audio_dim=128, num_classes=21):
    """
    test set loader that yields batches of random data.
    Replace this function with your real test dataset loader.
    """
    import torch

    num_batches = num_samples // batch_size
    for _ in range(num_batches):
        vision_feat = torch.randn(batch_size, vision_dim)
        audio_feat  = torch.randn(batch_size, audio_dim)
        labels = torch.randint(0, num_classes, (batch_size,))
        yield vision_feat, audio_feat, labels


def evaluate(model, test_loader, device='cuda'):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for vision_feat, audio_feat, label in tqdm(test_loader, desc="Evaluating"):
            vision_feat = vision_feat.to(device)
            audio_feat = audio_feat.to(device)
            label = label.to(device)
            output = model(vision_feat, audio_feat)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(label.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    return y_true, y_pred, y_probs

def print_metrics(y_true, y_pred, y_probs, n_classes):
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro')
    f1_weight = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    # PR AUC: One-vs-rest
    y_true_bin = np.eye(n_classes)[y_true]
    pr_auc = average_precision_score(y_true_bin, y_probs, average='macro')

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 (macro/weighted): {f1_mac:.4f} / {f1_weight:.4f}")
    print(f"Precision (macro):   {precision:.4f}")
    print(f"Recall (macro):      {recall:.4f}")
    print(f"PR AUC (macro):      {pr_auc:.4f}")
    print(f"MCC:                 {mcc:.4f}")

if __name__ == "__main__":
    # ---- Load test data ----
    test_loader = load_test_dataset()  # Replace with DataLoader or your custom loader

    # ---- Choose your fusion model ----
    # Example: vision/audio features dimension, output classes
    vision_dim = 256
    audio_dim = 128
    num_classes = 21

    # Example for Hybrid Fusion with xLSTM audio
    model = HybridFusion(visual_dim=vision_dim, audio_dim=audio_dim, out_dim=num_classes, late_fusion=False)
    # model = TensorFusion(visual_dim=vision_dim, audio_dim=audio_dim, out_dim=num_classes)
    # model = FiLMFusion(visual_dim=vision_dim, audio_dim=audio_dim, out_dim=num_classes)

    # --- Load pretrained weights ---
    model.load_state_dict(torch.load('checkpoint_best.pth', map_location='cpu'))
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Run evaluation ----
    y_true, y_pred, y_probs = evaluate(model, test_loader, device=('cuda' if torch.cuda.is_available() else 'cpu'))
    print_metrics(y_true, y_pred, y_probs, n_classes=num_classes)
