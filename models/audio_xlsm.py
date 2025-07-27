# models/audio_xlstm.py
import torch
import torch.nn as nn

class xLSTMBranch(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 96, 128), output_dim=128):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dims[1], batch_first=True)
        self.lstm3 = nn.LSTM(input_dim, hidden_dims[2], batch_first=True)
        self.fc = nn.Linear(sum(hidden_dims), output_dim)
    def forward(self, x):
        # x: (B, T, F)
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(x)
        out3, _ = self.lstm3(x)
        feat = torch.cat([out1[:, -1], out2[:, -1], out3[:, -1]], dim=-1)
        return self.fc(feat)

# Example usage:
# model = xLSTMBranch(input_dim=20, output_dim=128)
