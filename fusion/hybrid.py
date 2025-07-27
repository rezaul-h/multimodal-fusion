import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFusion(nn.Module):
    """
    Hybrid Fusion Module: Combines early (feature-level) and late (decision-level) fusion.
    - Early fusion: Concatenate visual and audio features, followed by joint MLP.
    - Late fusion (optional): Can be added as averaging/logit weighting if desired.
    """
    def __init__(self, visual_dim, audio_dim, hidden_dim=256, out_dim=21, dropout=0.2, late_fusion=False):
        super(HybridFusion, self).__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.joint_dim = hidden_dim
        self.out_dim = out_dim
        self.late_fusion = late_fusion

        # Early fusion: concatenate and project
        self.early_fusion = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        if late_fusion:
            # Late fusion: separate classifiers for vision/audio
            self.visual_classifier = nn.Linear(visual_dim, out_dim)
            self.audio_classifier = nn.Linear(audio_dim, out_dim)
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # Learnable fusion weight

    def forward(self, visual, audio):
        # Early fusion: feature concatenation
        fused_feat = torch.cat([visual, audio], dim=-1)
        early_out = self.early_fusion(fused_feat)
        if self.late_fusion:
            visual_out = self.visual_classifier(visual)
            audio_out = self.audio_classifier(audio)
            # Weighted average (learnable or fixed)
            fusion_w = torch.sigmoid(self.fusion_weight)
            late_out = fusion_w * visual_out + (1 - fusion_w) * audio_out
            # Combine early and late fusion (e.g., average, sum, or more sophisticated method)
            final_out = 0.6 * early_out + 0.4 * late_out  # Weighted sum; tune weights as needed
            return final_out
        else:
            return early_out

if __name__ == "__main__":
    vision = torch.randn(B, V)
    audio = torch.randn(B, A)
    # Only early fusion
    model = HybridFusion(visual_dim=V, audio_dim=A, out_dim=21, late_fusion=False)
    logits = model(vision, audio)
    print("Logits shape (early fusion):", logits.shape)
    # With late fusion enabled
    model_late = HybridFusion(visual_dim=V, audio_dim=A, out_dim=21, late_fusion=True)
    logits2 = model_late(vision, audio)
    print("Logits shape (hybrid early+late fusion):", logits2.shape)
