import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMFusion(nn.Module):
    """
    FiLM-based Attention Fusion Layer.
    Modulates visual features by conditioning on audio features using FiLM.
    Suitable for multimodal tasks (vision-audio, etc).
    """
    def __init__(self, visual_dim, audio_dim, hidden_dim=128, out_dim=None, dropout=0.1):
        super(FiLMFusion, self).__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim or visual_dim

        # Project audio features to scale and shift for FiLM
        self.film_gen = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim * 2)
        )
        # Output fusion
        self.out_proj = nn.Linear(visual_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual, audio):
        """
        visual: (B, visual_dim) or (B, T, visual_dim)
        audio:  (B, audio_dim) or (B, T, audio_dim)
        Returns: (B, out_dim)
        """
        # If input is sequence, flatten batch+T
        orig_shape = visual.shape
        is_seq = (visual.dim() == 3)
        if is_seq:
            B, T, D = visual.shape
            visual = visual.view(-1, D)
            audio = audio.view(-1, audio.shape[-1])

        # Generate FiLM parameters (scale, shift)
        gamma_beta = self.film_gen(audio)  # (B, 2*visual_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each (B, visual_dim)
        # Apply FiLM: scale and shift
        modulated = gamma * visual + beta
        modulated = self.dropout(F.relu(modulated))
        out = self.out_proj(modulated)

        if is_seq:
            out = out.view(B, T, -1)
        return out

if __name__ == "__main__":
    # Example usage
    B, V, A = 8, 256, 128
    visual = torch.randn(B, V)
    audio = torch.randn(B, A)
    film_fusion = FiLMFusion(visual_dim=V, audio_dim=A, out_dim=128)
    fused = film_fusion(visual, audio)
    print("Output shape:", fused.shape)  # (B, out_dim)
