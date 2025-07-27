# scripts/ablation_study.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

# --- Imports (update as per your repo structure) ---
from models.vision_xception import MiniXception
from models.audio_lstm import AudioLSTM
from models.audio_xlstm import xLSTMBranch
from models.fusion_hybrid import HybridFusion
from models.fusion_film import FiLMFusion

from training.train import MultiModalClassifier, MultiModalFeatureDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_CSV = './ablation_results.csv'

# --- Configurations to test ---
AUDIO_MODELS = ['lstm', 'xlstm']
FUSION_TYPES = ['hybrid', 'film']  # 'film' = FiLM (Attention-based)

def evaluate_model(model, loader, device, num_classes=21):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        pr_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception:
        pr_auc = float('nan')
    return acc, f1, pr_auc, mcc

def ablation_experiment(audio_model, fusion_type, train_loader, val_loader, feature_dims, num_classes=21):
    # --- Build model ---
    image_dim, audio_dim = feature_dims
    model_name = f"{fusion_type}_{audio_model}"
    print(f"\n=== Running {model_name.upper()} ===")
    # Select fusion model
    if fusion_type == 'hybrid':
        fusion = HybridFusion
    elif fusion_type == 'film':
        fusion = FiLMFusion
    else:
        raise ValueError("Invalid fusion type")
    # Select audio model branch
    if audio_model == 'lstm':
        audio_branch = AudioLSTM
    elif audio_model == 'xlstm':
        audio_branch = xLSTMBranch
    else:
        raise ValueError("Invalid audio model")

    # Wrap into your MultiModalClassifier or similar structure
    model = MultiModalClassifier(
        image_dim=image_dim,
        audio_dim=audio_dim,
        fusion=f"{fusion_type}_{audio_model}",
        out_dim=128,
        n_classes=num_classes
    ).to(DEVICE)
    # TODO: Optionally load specific checkpoint per run:
    # model.load_state_dict(torch.load(f"./checkpoints/{model_name}.pth", map_location=DEVICE))

    # --- Train model or load trained weights here ---
    # For brevity, this assumes you've trained and just want to evaluate
    # Otherwise, insert training loop here

    # --- Evaluate ---
    acc, f1, pr_auc, mcc = evaluate_model(model, val_loader, DEVICE, num_classes)
    print(f"Results: Acc={acc:.3f}, F1={f1:.3f}, PR_AUC={pr_auc:.3f}, MCC={mcc:.3f}")
    return {
        'Model': model_name,
        'Accuracy': acc,
        'F1': f1,
        'PR_AUC': pr_auc,
        'MCC': mcc
    }

def main():
    # --- Data Loaders ---
    train_csv = './data/splits/train_balanced.csv'
    val_csv = './data/splits/val.csv'
    train_loader = torch.utils.data.DataLoader(MultiModalFeatureDataset(train_csv, './data/features_balanced/train'), batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(MultiModalFeatureDataset(val_csv, './data/features_balanced/val'), batch_size=64, shuffle=False)
    # Feature dims inference (assuming concatenated features)
    sample, _ = next(iter(train_loader))
    total_dim = sample.shape[1]
    image_dim = audio_dim = total_dim // 2

    # --- Run ablation experiments ---
    results = []
    for fusion_type in FUSION_TYPES:
        for audio_model in AUDIO_MODELS:
            row = ablation_experiment(audio_model, fusion_type, train_loader, val_loader, (image_dim, audio_dim))
            results.append(row)

    # --- Save table ---
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(df)

    # --- Print LaTeX ---
    print("\nLaTeX Table:\n", df.to_latex(index=False, float_format="%.3f"))

    # --- Plot ---
    import matplotlib.pyplot as plt
    metrics = ['Accuracy', 'F1', 'PR_AUC', 'MCC']
    df_melted = df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)
    plt.title('Ablation Study: Model Variant Comparison')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
