import torch
from torch import nn
from torch.nn import functional as F

from models.resnet import resnet34, resnet50, resnet101
from models.Block import Block


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)  # 使用 1x1 卷积代替全连接层
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  # 使用 1x1 卷积代替全连接层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling
        se = self.global_avg_pool(x)  # 输出形状: (batch, channels, 1, 1)

        # 两个 1x1 卷积用于通道注意力计算
        se = self.fc1(se)  # 输出形状: (batch, channels // reduction, 1, 1)
        se = self.relu(se)
        se = self.fc2(se)  # 输出形状: (batch, channels, 1, 1)
        se = self.sigmoid(se)  # 输出形状: (batch, channels, 1, 1)

        # 对输入特征按通道加权
        return x * se  # 广播机制自动处理维度

class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling, ASPP, for high-level features
    """
    def __init__(self, in_channels=2048, mid_channels=512,out_channels=256,num_scales=6):
        super(ASPP, self).__init__() # original brance [0,6,12,18]
        # reduce dimensionality (1024 -> 512)
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.aspp1 = nn.Sequential(
            Block(mid_channels, out_channels, kernel=1, stride=1, padding=0, dilation=1),
            SEBlock(256)
        )

        self.aspp2 = nn.Sequential(
            Block(mid_channels, out_channels, kernel=3, stride=1, padding=2, dilation=2),
            SEBlock(256)
        )

        self.aspp3 = nn.Sequential(
            Block(mid_channels, out_channels, kernel=3, stride=1, padding=6, dilation=6),
            SEBlock(256)
        )

        self.aspp4 = nn.Sequential(
            Block(mid_channels, out_channels, kernel=3, stride=1, padding=12, dilation=12),
            SEBlock(256)
        )

        self.aspp5 = nn.Sequential(
            Block(mid_channels, out_channels, kernel=3, stride=1, padding=24, dilation=24),
            SEBlock(256)
        )

        self.aspp6 = nn.Sequential(
            Block(mid_channels, out_channels, kernel=3, stride=1, padding=36, dilation=36),
            SEBlock(256)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)  # reduce dimensionality

        # Scale Attention module: not used
        self.scale_attention = nn.Sequential(
            nn.Conv2d(mid_channels, 6, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.aspp8 = Block(out_channels * 7, out_channels, kernel=1)
        self.smooth = Block(out_channels, out_channels, kernel=3, padding=1, drop_out=True)

    def forward(self, x):
        x = self.reduce_channels(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp5(x)
        x6 = self.aspp6(x)

        x7 = self.global_avg_pool(x)
        x7 = self.conv1x1(x7)
        x7 = F.interpolate(x7, (64, 64), mode='bilinear')

        # fuse all channels
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)
        x = self.aspp8(x)
        x = self.smooth(x)

        return x

class CoralLab(nn.Module):
    def __init__(self, backbone_type='resnet34', out_channels=2):
        super(CoralLab, self).__init__()

        # Coral-Lab's backbone is resnet34, please do not change (resnet50 and 101 are for debugging)
        if backbone_type == 'resnet34':
            self.backbone = resnet34()
            hlf_ch = 512
            f1_ch = 64
            f2_ch = 128
            llf_ch = 64
        elif backbone_type == 'resnet50':
            self.backbone = resnet50()
            hlf_ch = 2048
            f1_ch = 256
            f2_ch = 512
            llf_ch = 64
        elif backbone_type == 'resnet101':
            self.backbone = resnet101()
            hlf_ch = 2048
            f1_ch = 256
            f2_ch = 512
            llf_ch = 64
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # ASPP module
        self.aspp = ASPP(in_channels=hlf_ch, mid_channels=512, out_channels=256)
        self.dropout1 = nn.Dropout(0.2)

        # llf processing
        self.conv1 = Block(llf_ch, 32, kernel=1)

        # f1 processing
        if f1_ch >64:
            self.convf1_1 = Block( f1_ch, 128, kernel=1)
            self.convf1_2 = Block(128,  64, kernel=1)
        else:
            self.convf1_1 = Block(64, 64, kernel=1)
            self.convf1_2 = Block(64, 64, kernel=1)


        # f2 processing
        if f2_ch > 128:
            self.convf2_1 = Block(f2_ch, 256, kernel=1)
            self.convf2_2 = Block(256, 128, kernel=1)
        else:
            self.convf2_1 = Block(128, 128, kernel=1)
            self.convf2_2 = Block(128, 128, kernel=1)

        # fuse processed llf,f1,f2,hlf and smooth
        self.convs2 = nn.Sequential(
            Block(256 + 32 + 64 + 128, 256, kernel=3, padding=1, drop_out=True),
            Block(256, 64, kernel=3, padding=1, drop_out=True),
            Block(64, 32, kernel=3, padding=1, drop_out=True),
        )

        # seg head
        self.conv3 = nn.Conv2d(32, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hlf, llf, f1, f2 = self.backbone(x)

        hlf = self.aspp(hlf)
        hlf = self.dropout1(hlf)
        hlf = F.interpolate(hlf, scale_factor=2, mode='bilinear', align_corners=True)

        llf = self.conv1(llf)

        f1 = self.convf1_1(f1)
        f1 = self.convf1_2(f1)

        f2 = self.convf2_1(f2)
        f2 = self.convf2_2(f2)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat((hlf, llf, f1, f2), dim=1)

        x = self.convs2(x)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.sigmoid(x)

        return x