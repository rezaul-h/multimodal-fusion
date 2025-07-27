# models/audio_lstm.py
import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=128, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        # Use last timestep (or mean-pool): 
        feat = out[:, -1]
        return self.fc(feat)

# Example usage:
# model = AudioLSTM(input_dim=20, output_dim=128)
