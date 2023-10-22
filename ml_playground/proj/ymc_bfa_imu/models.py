from torch import nn

from ml_playground.models.mstcn import MultiStageModel


class MstcnBinaryClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = MultiStageModel(
            num_stages=3,
            num_layers=9,
            in_feat_dim=18,
            num_f_maps=64,
            bottleneck_dim=48,
            out_feat_dim=128,
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        """(B, 18, T) -> (B, 2)"""
        out = self.encoder(x)
        out = self.fc(out)
        return out
