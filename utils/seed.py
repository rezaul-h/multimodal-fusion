import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, PyTorch (CPU/GPU), and CuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Example usage:
if __name__ == "__main__":
    set_seed(42)
    print("Random seed set!")
