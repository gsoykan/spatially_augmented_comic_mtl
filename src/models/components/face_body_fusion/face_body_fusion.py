import torch
import torch.nn as nn


class FaceBodyFusion(nn.Module):
    def __init__(self, input_dim):
        super(FaceBodyFusion, self).__init__()

        # Define layers for fusion
        self.fc1 = nn.Linear(input_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, input_dim)
        self.relu = nn.ReLU()

    def forward(self, fused_emb):
        fused_emb = self.relu(self.fc1(fused_emb))
        fused_emb = self.relu(self.fc2(fused_emb))
        fused_emb = self.relu(self.fc3(fused_emb))
        fused_emb = self.fc4(fused_emb)
        return fused_emb
