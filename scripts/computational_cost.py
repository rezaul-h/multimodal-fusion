# scripts/computational_cost.py

import torch
import numpy as np
import pandas as pd
import time
import gc
from thop import profile
from models.vision_xception import MiniXception
from models.audio_lstm import AudioLSTM
from models.audio_xlstm import xLSTMBranch
from models.fusion_hybrid import HybridFusion
from models.fusion_tensor import TensorFusion
from models.fusion_film import FiLMFusion
from training.train import MultiModalClassifier
from evaluation.evaluate import MultiModalFeatureDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define variants
FUSION_VARIANTS = [
    ("tensor", "lstm"),
    ("hybrid", "lstm"),
    ("hybrid", "xlstm"),
    ("film", "lstm"),
    ("film", "xlstm"),
]
VARIANT_NAMES = [
    "Tensor Fusion",
    "Hybrid Fusion (LSTM)",
    "Hybrid Fusion (xLSTM)",
    "Attention Fusion (LSTM)",
    "Attention Fusion (xLSTM)"
]
CHECKPOINTS = [
    './checkpoints/tensor_lstm.pth',
    './checkpoints/hybrid_lstm.pth',
    './checkpoints/hybrid_xlstm.pth',
    './checkpoints/film_lstm.pth',
    './checkpoints/film_xlstm.pth',
]
CONVERGED_EPOCHS = [23, 21, 18, 19, 17]  # Example values; update with your logs

# Dummy dataset to estimate inference/training speed and memory
TEST_CSV = './data/splits/test.csv'
FEATURE_DIR = './data/features_balanced/test'
BATCH_SIZE = 64

def measure_flops_params(model, input_shape):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(DEVICE)
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = 2 * macs  # MACs to FLOPs
    return params / 1e6, flops / 1e9  # Millions, Billions

def measure_time_and_memory(model, loader, train_mode=False):
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    if train_mode:
        model.train()
    times = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            start = time.time()
            _ = model(x)
            times.append(time.time() - start)
            if i > 2: break  # Only a few batches needed
    avg_time = np.mean(times[1:]) / x.shape[0] * 1000  # ms/sample
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    torch.cuda.empty_cache(); gc.collect()
    return avg_time, mem

def main():
    # --- Prepare Data ---
    ds = MultiModalFeatureDataset(TEST_CSV, FEATURE_DIR)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    feat_sample, _ = next(iter(loader))
    feat_dim = feat_sample.shape[1]
    image_dim = audio_dim = feat_dim // 2

    results = []
    for i, ((fusion, audio), variant_name, ckpt, converged_epoch) in enumerate(zip(FUSION_VARIANTS, VARIANT_NAMES, CHECKPOINTS, CONVERGED_EPOCHS)):
        print(f"Evaluating {variant_name}...")
        # Build model
        model = MultiModalClassifier(image_dim, audio_dim, fusion=f"{fusion}_{audio}", out_dim=128, n_classes=21).to(DEVICE)
        if ckpt and os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        # Params & FLOPs
        params, flops = measure_flops_params(model, input_shape=(BATCH_SIZE, feat_dim))
        # Training time/epoch (dummy timing, replace with real training logs if available)
        train_times = []
        for _ in range(3):
            t0 = time.time()
            x = torch.randn(BATCH_SIZE, feat_dim).to(DEVICE)
            y = torch.randint(0, 21, (BATCH_SIZE,)).to(DEVICE)
            logits = model(x)
            loss = torch.nn.CrossEntropyLoss()(logits, y)
            loss.backward(); model.zero_grad()
            train_times.append(time.time() - t0)
        train_time = np.mean(train_times)  # seconds per batch
        train_time_per_epoch = train_time * (len(ds) // BATCH_SIZE)
        # Inference time & memory
        infer_time, gpu_mem = measure_time_and_memory(model, loader)
        results.append({
            'Fusion Model': variant_name,
            'Params (M)': round(params, 2),
            'FLOPs (G)': round(flops, 2),
            'Train Time/Epoch (s)': round(train_time_per_epoch, 1),
            'Infer Time (ms)': round(infer_time, 2),
            'GPU Memory (MB)': round(gpu_mem, 1),
            'Converged Epoch': converged_epoch
        })
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('./evaluation/computational_cost.csv', index=False)
    print("\nLaTeX Table:\n", df.to_latex(index=False))

if __name__ == '__main__':
    main()
