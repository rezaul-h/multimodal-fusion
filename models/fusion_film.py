# models/fusion_film.py
import torch
import torch.nn as nn

class FiLMFusion(nn.Module):
    def __init__(self, v_dim, a_dim, out_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(a_dim, v_dim)
        self.beta_fc = nn.Linear(a_dim, v_dim)
        self.fc = nn.Linear(v_dim, out_dim)
    def forward(self, v, a):
        # v: (B, v_dim), a: (B, a_dim)
        gamma = self.gamma_fc(a)
        beta = self.beta_fc(a)
        mod_v = v * (1 + gamma) + beta  # FiLM modulation
        return self.fc(mod_v)

# Example usage:
# fusion = FiLMFusion(v_dim=128, a_dim=128, out_dim=128)
