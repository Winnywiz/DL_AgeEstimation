import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=strides)
        self.conv2      = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)

        if use_1x1conv:
            self.conv3  = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3  = None
        self.bn1        = nn.BatchNorm2d(num_features=out_channels)
        self.bn2        = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out     = self.conv1(x)
        out     = self.bn1(out)
        out     = F.relu(out)

        out     = self.conv2(out)
        out     = self.bn2(out)
        if self.conv3:
            x   = self.conv3(x)
        out += x
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, arch: tuple, in_channels, out_channels, num_features, kernel_size=3, strides=2):
        """
        Args:
            arch: Architecture block (num_residuals, in_channels, out_channels) e.g. arch=((2, 3, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512))
        """
        super(ResNet, self).__init__()
        self.arch           = arch
        self.conv2d         = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=3) # Input Layer
        self.bn             = nn.BatchNorm2d(num_features=out_channels)
        self.pool           = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 3*3 Max Pooling
        for i, b in enumerate(arch):
            self.add_module(f'b{i+2}', self.block(*b, first_block=(i==0))) # Add Residual Block
        self.gap            = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten        = nn.Flatten()
        self.linear         = nn.LazyLinear(out_features=num_features)

    def block(self, num_residuals, in_channels, out_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0:
                do_1x1 = (not first_block) or (in_channels != out_channels)
                blk.append(Residual(in_channels, out_channels, kernel_size=3, use_1x1conv=do_1x1, strides=2 if not first_block else 1))
            else:
                blk.append(Residual(out_channels, out_channels, kernel_size=3))
        return nn.Sequential(*blk)

    def forward(self, x):
        out = self.conv2d(x)    
        out = self.bn(out)
        out = F.relu(out)
        out = self.pool(out)

        for i in range(len(self.arch)):
            out = getattr(self, f'b{i+2}')(out)
        out = self.gap(out)
        out = self.flatten(out)
        return self.linear(out)