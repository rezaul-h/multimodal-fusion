# models/fusion_hybrid.py
import torch
import torch.nn as nn

class HybridFusion(nn.Module):
    def __init__(self, v_dim, a_dim, out_dim):
        super().__init__()
        # Early fusion: concatenate then fc
        self.fc_early = nn.Linear(v_dim + a_dim, out_dim)
        # Late fusion: process separately, then combine
        self.fc_v = nn.Linear(v_dim, out_dim)
        self.fc_a = nn.Linear(a_dim, out_dim)
        self.fc_late = nn.Linear(out_dim * 2, out_dim)
    def forward(self, v, a):
        early = self.fc_early(torch.cat([v, a], dim=-1))
        v_ = self.fc_v(v)
        a_ = self.fc_a(a)
        late = self.fc_late(torch.cat([v_, a_], dim=-1))
        return (early + late) / 2  # or other fusion strategy

# Example usage:
# fusion = HybridFusion(v_dim=128, a_dim=128, out_dim=128)
