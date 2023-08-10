#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels , in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)  #!20220104
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
#%%
class UNet(nn.Module):
    def __init__(self,in_channels=2, c1=16):
        super(UNet, self).__init__()
        self.max_pool = nn.MaxPool1d(2)
        self.down_conv1 = DoubleConv(in_channels, c1)
        self.down_conv2 = DoubleConv(c1, c1*2)
        self.down_conv3 = DoubleConv(c1*2, c1*4)
        self.bottom_conv = DoubleConv(c1*4, c1*8)
        self.up_conv1 = Up(c1*8, c1*4)
        self.up_conv2 = Up(c1*4, c1*2)
        self.up_conv3 = Up(c1*2, c1)
        self.out_conv = nn.Conv1d(c1, 1, kernel_size=3, padding=1, padding_mode='circular')

    def forward(self, x):
        x1 = self.down_conv1(x)
        x = self.max_pool(x1)
        x2 = self.down_conv2(x)
        x = self.max_pool(x2)
        x3 = self.down_conv3(x)
        x = self.max_pool(x3)
        x = self.bottom_conv(x)
        x = self.up_conv1(x, x3)
        x = self.up_conv2(x, x2)
        x = self.up_conv3(x, x1)
        x = self.out_conv(x)
        return x
        pass
#%%
if __name__ == '__main__':
    net = UNet()
    x = torch.rand((3,2,64))
    print(net(x).size())
# %%
