import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorFusion(nn.Module):
    """
    Tensor Fusion Network (TFN) Layer for Multimodal Fusion.
    Computes the outer product between visual and audio features for each sample,
    then flattens and projects to the output space.
    Reference: Zadeh et al. "Tensor Fusion Network for Multimodal Sentiment Analysis", EMNLP 2017.
    """
    def __init__(self, visual_dim, audio_dim, hidden_dim=256, out_dim=21, dropout=0.2):
        super(TensorFusion, self).__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.fused_dim = (visual_dim + 1) * (audio_dim + 1)
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, visual, audio):
        # visual: (B, Dv), audio: (B, Da)
        B = visual.size(0)
        # Add bias term (as in original TFN paper)
        visual_ = torch.cat([visual, visual.new_ones(B, 1)], dim=1)  # (B, Dv+1)
        audio_  = torch.cat([audio,  audio.new_ones(B, 1)], dim=1)   # (B, Da+1)
        # Outer product: (B, Dv+1, Da+1)
        outer = torch.bmm(visual_[:, :, None], audio_[:, None, :])   # (B, Dv+1, Da+1)
        fusion_feat = outer.view(B, -1)  # Flatten to (B, (Dv+1)*(Da+1))
        out = self.proj(fusion_feat)     # Project to output
        return out

if __name__ == "__main__":
    visual = torch.randn(B, V)
    audio = torch.randn(B, A)
    model = TensorFusion(visual_dim=V, audio_dim=A, out_dim=21)
    logits = model(visual, audio)
    print("Logits shape:", logits.shape)
