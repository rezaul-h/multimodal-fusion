# evaluation/confusion_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix

PRED_CSV = './evaluation/predictions.csv'
SAVE_FIG = './evaluation/classwise_error.png'

def analyze_confusion(pred_csv, save_fig=None):
    df = pd.read_csv(pred_csv)
    true_labels = df['true'].tolist()
    pred_labels = df['pred'].tolist()
    class_names = sorted(list(set(true_labels) | set(pred_labels)))
    n_classes = len(class_names)
    class2idx = {c: i for i, c in enumerate(class_names)}
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    errors = support - correct
    error_rate = 100. * errors / support
    accuracy = 100. * correct / support

    # Per-class results
    per_class = pd.DataFrame({
        'Class': class_names,
        'Support': support,
        'Correct': correct,
        'Error (%)': error_rate.round(2),
        'Accuracy (%)': accuracy.round(2)
    }).sort_values('Error (%)', ascending=False)

    print(per_class.to_string(index=False))
    print("\nEasiest class:", per_class.iloc[-1]['Class'], "Hardest:", per_class.iloc[0]['Class'])

    # Bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Error (%)', data=per_class, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Error Percentage (%)')
    plt.title('Class-wise Error Rates')
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()
    
    # Top confusions per class
    print("\nTop Confusions per Class:")
    for i, cls in enumerate(class_names):
        most_confused_idx = cm[i].copy()
        most_confused_idx[i] = 0  # zero the diagonal
        if most_confused_idx.sum() == 0:
            continue
        j = np.argmax(most_confused_idx)
        confused_class = class_names[j]
        print(f"  {cls:20s} → {confused_class:20s} ({cm[i,j]} times)")

if __name__ == '__main__':
    analyze_confusion(PRED_CSV, save_fig=SAVE_FIG)
