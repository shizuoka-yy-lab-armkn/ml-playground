import torch.nn.functional as F
from torch import Tensor, nn


class MultiStageModel(nn.Module):
    """
    (B, in_feat_dim, T) -> (B, bottleneck_dim)
    """

    def __init__(
        self,
        num_stages: int,
        num_layers: int,
        num_f_maps: int,
        in_feat_dim: int,
        bottleneck_dim: int,
        out_feat_dim: int,
    ):
        super().__init__()
        self.stage1 = SingleStageModel(
            num_layers, num_f_maps, in_feat_dim, bottleneck_dim
        )
        self.stages = nn.ModuleList(
            [
                SingleStageModel(num_layers, num_f_maps, bottleneck_dim, bottleneck_dim)
                for _ in range(num_stages - 1)
            ]
        )
        self.stage_z = SingleStageModel(
            num_layers, num_f_maps, bottleneck_dim, out_feat_dim
        )

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        # (B, in_feat_dim, T) -> (B, bottleneck_dim, T)
        out = self.stage1(x)

        # -> (B, bottleneck_dim, T)
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))

        # -> (B, out_feat_dim, T)
        out = self.stage_z(F.softmax(out, dim=1))

        out: Tensor = self.avgpool(out)
        out = out.squeeze(dim=-1)

        return out


class SingleStageModel(nn.Module):
    """
    (B, in_feat_dim, T) -> (B, bottleneck_dim, T)
    """

    def __init__(
        self, num_layers: int, num_f_maps: int, in_feat_dim: int, bottleneck_dim: int
    ):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(in_feat_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2**i, num_f_maps) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, bottleneck_dim, 1)

    def forward(self, x):
        # (B, in_feat_dim, T) -> (B, num_f_maps, T)
        out = self.conv_1x1(x)

        # shape doesn't change (B, num_f_maps, T)
        for layer in self.layers:
            out = layer(out)

        # (B, num_f_maps, T) -> (B, bottleneck_dim, T)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    """
    (B, C, T) -> (B, C, T)
    input.size() == output.size()
    """

    def __init__(self, dilation: int, inout_channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            inout_channels,
            inout_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(inout_channels, inout_channels, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
