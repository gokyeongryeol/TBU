import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .module import ResNet, WideResNet


class Architecture(BaseModel):
    def __init__(
        self,
        arc,
        dp_rate=0.0,
        n_classes=None,
        use_thres=False,
        unlabel_size=None,
        n_filter=0,
    ):
        super().__init__()

        arc_cfg = arc.split("-")
        if arc_cfg[0] == "res":
            depth = arc_cfg[1]
            self.emb_net = ResNet(int(depth), num_classes=0)
        elif arc_cfg[0] == "wrn":
            depth, width = arc_cfg[1], arc_cfg[2]
            self.emb_net = WideResNet(
                int(depth), int(width), dropout_rate=0.0, num_classes=0
            )
        else:
            raise NotImplementedError

        self.arc = arc
        self.embDim = self.emb_net.num_features
        self.dp_rate = dp_rate
        self.n_classes = n_classes
        self.linear = nn.Linear(self.embDim, n_classes)

        if use_thres:
            self.register_buffer("ema_g", torch.tensor(1 / n_classes))
            self.register_buffer("ema_l", torch.ones(n_classes) / n_classes)
            self.register_buffer("ema_h", torch.ones(n_classes) / n_classes)
            if n_filter > 0:
                self.register_buffer(
                    "hard_idxs", torch.ones(unlabel_size, dtype=torch.bool)
                )
            else:
                self.register_buffer(
                    "hard_idxs", torch.zeros(unlabel_size, dtype=torch.bool)
                )

    def forward_features(self, x):
        emb = self.emb_net(x)
        return emb

    def forward_head(self, emb):
        if self.dp_rate > 0:
            emb = F.dropout(emb, p=self.dp_rate, training=True)

        out = self.linear(emb)
        return out
