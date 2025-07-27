# scripts/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_CSV = './ablation_results.csv'
COST_CSV = './evaluation/computational_cost.csv'
PLOT_DIR = './evaluation/plots'

def plot_metrics(results_csv, save_dir):
    df = pd.read_csv(results_csv)
    metrics = ['Accuracy', 'F1', 'PR_AUC', 'MCC']
    model_names = df['Model'] if 'Model' in df.columns else df['Fusion Model']
    # Grouped barplot
    plt.figure(figsize=(10, 6))
    df_plot = df.copy()
    df_plot['Model'] = model_names
    df_plot = df_plot.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_plot)
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha='right')
    plt.title('Model Performance Metrics')
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(f'{save_dir}/model_performance_bar.png', dpi=300)
    plt.show()

def plot_cost(cost_csv, save_dir):
    df = pd.read_csv(cost_csv)
    # Cost metrics to plot
    cost_metrics = ['Params (M)', 'FLOPs (G)', 'Infer Time (ms)', 'GPU Memory (MB)', 'Converged Epoch']
    for metric in cost_metrics:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Fusion Model', y=metric, data=df, palette='crest')
        plt.xticks(rotation=15, ha='right')
        plt.title(f'Computational Cost: {metric}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/cost_{metric.replace(" ", "_").lower()}.png', dpi=300)
        plt.show()
    # Radar chart (optional: for all cost metrics normalized)
    try:
        from math import pi
        import numpy as np
        radar_df = df.set_index('Fusion Model')[cost_metrics]
        # Normalize for radar
        norm_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-8)
        labels = radar_df.index.tolist()
        num_vars = len(cost_metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)] + [0]
        plt.figure(figsize=(7, 7))
        for i, (idx, row) in enumerate(norm_df.iterrows()):
            values = row.tolist() + [row.tolist()[0]]
            plt.polar(angles, values, marker='o', label=labels[i])
        plt.xticks(angles[:-1], cost_metrics, color='grey', size=10)
        plt.yticks([0.25, 0.5, 0.75], ['0.25','0.5','0.75'], color='grey', size=8)
        plt.title('Normalized Computational Cost Radar Chart')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
        plt.tight_layout()
        plt.savefig(f'{save_dir}/cost_radar.png', dpi=300)
        plt.show()
    except Exception as e:
        print("Radar chart plotting failed:", e)

if __name__ == '__main__':
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Plotting model performance metrics...")
    plot_metrics(RESULTS_CSV, PLOT_DIR)
    print("Plotting computational cost metrics...")
    plot_cost(COST_CSV, PLOT_DIR)
