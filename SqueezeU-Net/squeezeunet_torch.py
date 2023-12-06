import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels // 2, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.expand1x1(x)), F.relu(self.expand3x3(x))], 1)

class SqueezeUNet(nn.Module):
    def __init__(self, num_classes = 1, channels_shrink_factor=1):
        super(SqueezeUNet, self).__init__()

        self.channels_shrink_factor = channels_shrink_factor

        # Contracting Path
        self.down1 = FireModule(1, int(16 / self.channels_shrink_factor), int(32 / self.channels_shrink_factor))
        self.down2 = FireModule(int(32 / self.channels_shrink_factor), int(16 / self.channels_shrink_factor), int(64 / self.channels_shrink_factor))
        self.down3 = FireModule(int(64 / self.channels_shrink_factor), int(32 / self.channels_shrink_factor), int(128 / self.channels_shrink_factor))
        self.down4 = FireModule(int(128 / self.channels_shrink_factor), int(48 / self.channels_shrink_factor), int(256 / self.channels_shrink_factor))

        # Bottleneck
        self.bottleneck = FireModule(int(256 / self.channels_shrink_factor), int(64 / self.channels_shrink_factor), int(512 / self.channels_shrink_factor))

        # Expansive Path
        self.up4 = FireModule(int(512 / self.channels_shrink_factor) + int(256 / self.channels_shrink_factor), int(48 / self.channels_shrink_factor), int(256 / self.channels_shrink_factor))
        self.up3 = FireModule(int(256 / self.channels_shrink_factor) + int(128 / self.channels_shrink_factor), int(32 / self.channels_shrink_factor), int(128 / self.channels_shrink_factor))
        self.up2 = FireModule(int(128 / self.channels_shrink_factor) + int(64 / self.channels_shrink_factor), int(16 / self.channels_shrink_factor), int(64 / self.channels_shrink_factor))
        self.up1 = FireModule(int(64 / self.channels_shrink_factor) + int(32 / self.channels_shrink_factor), int(16 / self.channels_shrink_factor), int(32 / self.channels_shrink_factor))

        # Final convolution
        self.final_conv = nn.Conv2d(int(32 / self.channels_shrink_factor), num_classes, kernel_size=1)

    def forward(self, x):
        # x = x.unsqueeze(1)

        # Contracting Path
        x1 = self.down1(x)
        x1p = F.max_pool2d(x1, 2)
        x2 = self.down2(x1p)
        x2p = F.max_pool2d(x2, 2)
        x3 = self.down3(x2p)
        x3p = F.max_pool2d(x3, 2)
        x4 = self.down4(x3p)
        x4p = F.max_pool2d(x4, 2)

        # Bottleneck
        bn = self.bottleneck(x4p)

        # Expansive Path
        up4 = F.interpolate(bn, scale_factor=2, mode='bilinear', align_corners=True)
        up4 = torch.cat([up4, x4], dim=1)
        up4 = self.up4(up4)

        up3 = F.interpolate(up4, scale_factor=2, mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, x3], dim=1)
        up3 = self.up3(up3)

        up2 = F.interpolate(up3, scale_factor=2, mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, x2], dim=1)
        up2 = self.up2(up2)

        up1 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, x1], dim=1)
        up1 = self.up1(up1)

        # Final convolution
        out = self.final_conv(up1)
        return out
