from torch import nn
from config import *
from models.Block import Block

# input 512*512*3 RGB image, out put segmentation mask with 2 classes
class DownSample(nn.Module): # encode layer without dropout
    def __init__(self, inplanes, outplanes):
        super(DownSample, self).__init__()
        self.maxpooling = nn.MaxPool2d(2, stride=2)

        self.convs = nn.Sequential(Block(inplanes, outplanes, 3,
                                         padding=1, bias=True),
                                   Block(outplanes, outplanes, 3,
                                         padding=1, bias=True),
                                   )

    def forward(self, x):
        x = self.maxpooling(x)
        x = self.convs(x)
        return x

class UpSample(nn.Module): # decode layer without dropout
    def __init__(self, inplanes, outplanes):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convs = nn.Sequential(Block(inplanes, outplanes, 3,
                                         padding=1, bias=True,drop_out=False),
                                   Block(outplanes, outplanes, 3,
                                         padding=1, bias=True,drop_out=False),
                                   )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.convs(x)
        return x

# ================================================ U-Net Light version================================================ #
#
# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=2):
#         super(UNet, self).__init__()
#
#         # Modify the first layer to accept 3 input channels
#         self.convs = nn.Sequential(Block(in_plane=3, out_plane=64, kernel=3,
#                                          padding=1, bias=True),
#                                    Block(in_plane=64, out_plane=64, kernel=3,
#                                          padding=1, bias=True),
#                                    )
#         self.down1 = DownSample(64, 128)
#         self.down2 = DownSample(128, 256)
#         self.down3 = DownSample(256, 512)
#
#         self.up1 = UpSample(256 + 512, 256)
#         self.up2 = UpSample(128 + 256, 128)
#         self.up3 = UpSample(64 + 128, 64)
#
#         self.cls = nn.Conv2d(64, 2, 1)
#         self.sigmoid = nn.Sigmoid()  # Add sigmoid activation
#
#     def forward(self, x):
#         # encoder
#         x1 = self.convs(x)  # [B, 32, 512, 512]
#         x2 = self.down1(x1)  # [B, 64, 256, 256]
#         x3 = self.down2(x2)  # [B, 128, 128, 128]
#         x4 = self.down3(x3)  # [B, 256, 64, 64]
#
#         # decoder
#         x = self.up1(x4, x3)  # [B, 128, 128, 128]
#         x = self.up2(x, x2)  # [B, 64, 256, 256]
#         x = self.up3(x, x1)  # [B, 32, 512, 512]
#         x = self.cls(x)  # [B, out_channels, 512, 512]
#         x = self.sigmoid(x)  # Apply sigmoid activation
#
#         return x

# ================================================ U-Net from Prof.Mizuno ================================================ #
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()

        # Modify the first layer to accept 3 input channels
        self.convs = nn.Sequential(Block(in_plane=3, out_plane=64, kernel=3,
                                         padding=1, bias=True),
                                   Block(in_plane=64, out_plane=64, kernel=3,
                                         padding=1, bias=True),
                                   )
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)

        self.up1 = UpSample(512 + 1024, 512)
        self.up2 = UpSample(256 + 512, 256)
        self.up3 = UpSample(128 + 256, 128)
        self.up4 = UpSample(64 + 128, 64)

        self.cls = nn.Conv2d(64, 2, 1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        # encoder
        x1 = self.convs(x)   # [B,   64, 512, 512]
        x2 = self.down1(x1)  # [B,  128, 256, 256]
        x3 = self.down2(x2)  # [B,  256, 128, 128]
        x4 = self.down3(x3)  # [B,  512,  64,  64]
        x5 = self.down4(x4)  # [B, 1024,  32,  32]

        # decoder
        x = self.up1(x5, x4) # [B, 512,  64,  64]
        x = self.up2(x, x3)  # [B, 256, 128, 128]
        x = self.up3(x, x2)  # [B, 128, 256, 256]
        x = self.up4(x, x1)  # [B,  64, 512, 512]
        x = self.cls(x)      # [B,   2, 512, 512]
        x = self.sigmoid(x)  # Apply sigmoid activation

        return x