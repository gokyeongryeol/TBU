import torch.nn as nn


def cfg(depth):
    depth_lst = [11, 13, 16, 19]
    assert depth in depth_lst, "Error : VGGnet depth should be either 11, 13, 16, 19"
    cf_dict = {
        '11': [
            64, 'mp',
            128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'],
        '13': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'
            ],
        '16': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 'mp',
            512, 512, 512, 'mp',
            512, 512, 512, 'mp'
            ],
        '19': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 256, 'mp',
            512, 512, 512, 512, 'mp',
            512, 512, 512, 512, 'mp'
            ],
    }

    return cf_dict[str(depth)]


class VGG(nn.Module):
    def __init__(self, depth, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg(depth))
        self.num_features = 512

        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.num_features, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_planes = 3

        for x in cfg:
            if x == "mp":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_planes, x, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_planes = x

        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward_features(self, x):
        emb = self.features(x)
        emb = emb.view(emb.size(0), -1)
        return emb

    def forward_head(self, emb):
        out = self.fc(emb)
        return out

    def forward(self, x):
        emb = self.forward_features(x)
        out = self.forward_head(emb)
        return out
