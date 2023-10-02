import torch.nn as nn
import torch.nn.functional as F


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (
        depth in depth_lst
    ), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        "18": (BasicBlock, [2, 2, 2, 2]),
        "34": (BasicBlock, [3, 4, 6, 3]),
        "50": (Bottleneck, [3, 4, 6, 3]),
        "101": (Bottleneck, [3, 4, 23, 3]),
        "152": (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.num_features = 512 * block.expansion

        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.num_features, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        emb = F.relu(self.bn1(self.conv1(x)))
        emb = self.layer1(emb)
        emb = self.layer2(emb)
        emb = self.layer3(emb)
        emb = self.layer4(emb)
        emb = F.avg_pool2d(emb, 4)
        emb = emb.view(emb.size(0), -1)
        return emb

    def forward_head(self, emb):
        out = self.fc(emb)
        return out

    def forward(self, x):
        emb = self.forward_features(x)
        out = self.forward_head(emb)
        return out
