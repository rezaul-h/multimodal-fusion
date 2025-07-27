# evaluation/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_auc_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

def compute_all_metrics(y_true, y_pred, y_probs=None, n_classes=None):
    """Compute all required metrics for multiclass classification."""
    results = {}

    # Basic metrics
    results['Accuracy'] = accuracy_score(y_true, y_pred)
    results['F1'] = f1_score(y_true, y_pred, average='macro')
    results['MCC'] = matthews_corrcoef(y_true, y_pred)

    # Macro PR AUC
    if y_probs is not None and n_classes is not None:
        # One-hot encoding for true labels
        y_true_bin = np.zeros((len(y_true), n_classes))
        y_true_bin[np.arange(len(y_true)), y_true] = 1
        pr_aucs = []
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            pr_auc = np.trapz(precision, recall)
            pr_aucs.append(pr_auc)
        results['PR_AUC'] = np.mean(pr_aucs)
        # ROC AUC (optional)
        try:
            results['ROC_AUC'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception:
            results['ROC_AUC'] = np.nan
    else:
        results['PR_AUC'] = np.nan
        results['ROC_AUC'] = np.nan

    # Confusion Matrix & class-wise error
    cm = confusion_matrix(y_true, y_pred)
    results['Confusion_Matrix'] = cm
    error_per_class = (1 - np.diag(cm) / np.maximum(np.sum(cm, axis=1), 1)).round(4)
    results['Classwise_Error'] = error_per_class

    return results

def metrics_dataframe(results_dict, class_labels=None):
    """Convert metrics and class-wise errors to a DataFrame for reporting."""
    basic_metrics = ['Accuracy', 'F1', 'PR_AUC', 'MCC', 'ROC_AUC']
    metrics = {k: [results_dict[k]] for k in basic_metrics if k in results_dict}
    df_main = pd.DataFrame(metrics)
    # Class-wise error as DataFrame
    if 'Classwise_Error' in results_dict and class_labels is not None:
        err_df = pd.DataFrame({
            'Class': class_labels,
            'Classwise_Error': results_dict['Classwise_Error']
        })
        return df_main, err_df
    return df_main, None

def report_classification(y_true, y_pred, class_names=None):
    """Print full classification report and confusion matrix."""
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

def latex_table_from_df(df, floatfmt="%.4f"):
    """Export DataFrame to LaTeX format for paper-ready tables."""
    return df.to_latex(index=False, float_format=floatfmt)

def save_metrics_to_csv(df, filename):
    df.to_csv(filename, index=False)