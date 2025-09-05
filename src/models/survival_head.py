from torch import nn

class DeepSurvHead(nn.Module):
    def __init__(self, in_dim=768, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)
