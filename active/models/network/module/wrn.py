import torch.nn as nn
import torch.nn.functional as F


class WideBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._wide_layer(WideBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.num_features = nStages[3]

        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.num_features, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward_features(self, x):
        emb = self.conv1(x)
        emb = self.layer1(emb)
        emb = self.layer2(emb)
        emb = self.layer3(emb)
        emb = F.relu(self.bn1(emb))
        emb = F.avg_pool2d(emb, 8)
        emb = emb.view(emb.size(0), -1)
        return emb

    def forward_head(self, emb):
        out = self.fc(emb)
        return out

    def forward(self, x):
        emb = self.forward_features(x)
        out = self.forward_head(emb)
        return out
