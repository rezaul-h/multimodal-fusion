# models/fusion_tensor.py
import torch
import torch.nn as nn

class TensorFusion(nn.Module):
    def __init__(self, v_dim, a_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.fc = nn.Linear(v_dim * a_dim, out_dim)
    def forward(self, v, a):
        # v: (B, v_dim), a: (B, a_dim)
        outer = torch.bmm(v.unsqueeze(2), a.unsqueeze(1))  # (B, v_dim, a_dim)
        flat = outer.view(outer.size(0), -1)
        return self.fc(flat)

# Example usage:
# fusion = TensorFusion(v_dim=128, a_dim=128, out_dim=128)
# out = fusion(vision_feat, audio_feat)
