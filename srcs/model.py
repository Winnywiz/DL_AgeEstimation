import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, stride=strides) if use_1x1conv else None
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(out + x)


class ResNet(nn.Module):
    """
    ResNet built from scratch — used as a historical baseline to show
    why transfer learning from pretrained weights outperforms training
    from random initialization on limited data.
    """
    def __init__(self, arch: tuple, in_channels, out_channels, num_classes, kernel_size=3, strides=2):
        super().__init__()
        self.arch    = arch
        self.conv2d  = nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding=3)
        self.bn      = nn.BatchNorm2d(out_channels)
        self.pool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i, b in enumerate(arch):
            self.add_module(f"b{i+2}", self._make_block(*b, first_block=(i == 0)))
        self.gap     = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear  = nn.LazyLinear(out_features=num_classes)

    def _make_block(self, num_residuals, in_channels, out_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0:
                do_1x1 = (not first_block) or (in_channels != out_channels)
                blk.append(Residual(in_channels, out_channels, use_1x1conv=do_1x1,
                                    strides=2 if not first_block else 1))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        out = F.relu(self.bn(self.conv2d(x)))
        out = self.pool(out)
        for i in range(len(self.arch)):
            out = getattr(self, f"b{i+2}")(out)
        return self.linear(self.flatten(self.gap(out)))

def ResNet50_base(num_classes=4):
    """
    Pretrained ResNet-50 with frozen backbone and a single linear head.
    Used as the baseline model — no regularization techniques applied.
    """
    weights = ResNet50_Weights.DEFAULT
    model   = resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def ResNet50(num_classes=4, freeze_backbone=True):
    """
    Pretrained ResNet-50 with dropout + linear classification head.
    freeze_backbone=True  → Phase 1: head-only training
    freeze_backbone=False → Phase 2: full fine-tuning
    """
    weights = ResNet50_Weights.DEFAULT
    model   = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model, weights


def EfficientNetB0(num_classes=4, freeze_backbone=True):
    """
    Pretrained EfficientNet-B0 with dropout + linear classification head.
    freeze_backbone=True  → Phase 1: head-only training
    freeze_backbone=False → Phase 2: full fine-tuning
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model   = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model, weights
