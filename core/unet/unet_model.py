""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet(nn.Module):
    def __init__(
        self, 
        in_channels, out_channels, 
        bilinear=False,
        block_out_channels=[64, 128, 256, 512, 1024]
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.depth = len(block_out_channels)
        self.down_part = [None] * self.depth
        for i in range(len(block_out_channels)):
            if i == 0:
                self.down_part[i] = DoubleConv(
                    in_channels, block_out_channels[i]
                )
            elif i < (self.depth - 1):
                self.down_part[i] = Down(
                    block_out_channels[i - 1],
                    block_out_channels[i]
                )
            else:
                self.down_part[i] = Down(
                    block_out_channels[i - 1],
                    block_out_channels[i] // factor
                )
        self.down_part = nn.Sequential(*self.down_part)

        self.up_part = [None] * self.depth
        for i in range(self.depth - 1, -1, -1):
            if i == 0:
                self.up_part[i] = OutConv(
                    block_out_channels[i], 
                    out_channels,
                )
            elif i == 1:
                self.up_part[i] = Up(
                    block_out_channels[i],
                    block_out_channels[i - 1],
                    bilinear
                )
            else:
                self.up_part[i] = Up(
                    block_out_channels[i],
                    block_out_channels[i - 1] // factor,
                    bilinear
                )
            self.up_part = nn.Sequential(*self.up_part)

    def forward(self, x):
        out = [None] * self.depth
        for i in range(self.depth):
            if i == 0:
                out[i] = self.down_part[i](x)
            else:
                out[i] = self.down_part[i](out[i - 1])

        ret = None
        for i in range(self.depth - 1, -1, -1):
            if i == 0:
                ret = self.up_part[i](ret)
            elif i < (self.depth - 1):
                ret = self.up_part[i](ret, out[i - 1])
            else:
                ret = self.up_part[i](out[i], out[i - 1])
        
        return ret


class SimUNet(nn.Module):
    def __init__(
        self, 
        in_channels, out_channels, 
        kernels_per_layer=2,
        bilinear=True,
        block_out_channels=[64, 128, 256, 512, 1024]
    ):
        super(SimUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        factor = 2 if bilinear else 1

        self.depth = len(block_out_channels)
        self.down_part = [None] * self.depth
        for i in range(len(block_out_channels)):
            if i == 0:
                self.down_part[i] = DoubleConvDS(
                    in_channels, block_out_channels[i],
                    kernels_per_layer=kernels_per_layer
                )
            elif i < (self.depth - 1):
                self.down_part[i] = DownDS(
                    block_out_channels[i - 1],
                    block_out_channels[i], 
                    kernels_per_layer=kernels_per_layer
                )
            else:
                self.down_part[i] = DownDS(
                    block_out_channels[i - 1],
                    block_out_channels[i] // factor, 
                    kernels_per_layer=kernels_per_layer
                )
        self.down_part = nn.Sequential(*self.down_part)

        self.up_part = [None] * self.depth
        for i in range(self.depth - 1, -1, -1):
            if i == 0:
                self.up_part[i] = OutConv(
                    block_out_channels[i], 
                    out_channels,
                )
            elif i == 1:
                self.up_part[i] = UpDS(
                    block_out_channels[i], 
                    block_out_channels[i - 1], 
                    bilinear, 
                    kernels_per_layer=kernels_per_layer
                )
            else:
                self.up_part[i] = UpDS(
                    block_out_channels[i], 
                    block_out_channels[i - 1] // factor, 
                    bilinear, 
                    kernels_per_layer=kernels_per_layer
                )
            self.up_part = nn.Sequential(*self.up_part)

    def forward(self, x):
        out = [None] * self.depth
        for i in range(self.depth):
            if i == 0:
                out[i] = self.down_part[i](x)
            else:
                out[i] = self.down_part[i](out[i - 1])

        ret = None
        for i in range(self.depth - 1, -1, -1):
            if i == 0:
                ret = self.up_part[i](ret)
            elif i < (self.depth - 1):
                ret = self.up_part[i](ret, out[i - 1])
            else:
                ret = self.up_part[i](out[i], out[i - 1])
        
        return ret
