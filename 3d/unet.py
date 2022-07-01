#%%
#? last modified on 20211231
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
            #nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=2, padding_mode='circular'),    #!20211230      #?tentative
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='circular'),    #!20211230      #?tentative
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),    #!20211230
            #nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=2,padding_mode='circular'),    #!20211230      #?tentative
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),    #!20211230      #?tentative
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)    #!20211230
        )

    def forward(self, x):
        # print x.size()
        #?print("Middle: "+ str(x.size()))     #!20220102
        return self.double_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool1d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)
#%%
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        
        #self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)  #!20220102
        self.up = nn.ConvTranspose1d(in_channels , in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)  #!20220104
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):

        x1 = self.up(x1)
        #?print(x1.shape,x2.shape)         #!20220102
        # input()
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #x = torch.cat([x2, x1], dim=1)         #!20220102
        x = torch.cat([x2, x1], dim=1)         #!20220102
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

        #self.out_conv = nn.Conv1d(c1, 1, kernel_size=3, padding=2, padding_mode='circular')    #!20211230      #?tentative
        self.out_conv = nn.Conv1d(c1, 1, kernel_size=3, padding=1, padding_mode='circular')    #!20211230      #?tentative

    def forward(self, x):
        # print x.size()    #!20220102 
        #?print("initial; "+ str(x.size()))    #!20220102
        x1 = self.down_conv1(x)
        # print x1.size()    #!20220102 
        #?print("down_conv1; "+ str(x1.size()))    #!20220102
        x = self.max_pool(x1)
        # print x.size()    #!20220102 
        #?print("max_pool; "+ str(x.size()))    #!20220102
        x2 = self.down_conv2(x)
        #?print("down_conv2; "+ str(x2.size()))    #!20220102
        x = self.max_pool(x2)
        # print x.size()    #!20220102 
        #?print("max_pool; "+ str(x.size()))    #!20220102
        x3 = self.down_conv3(x)
        # print x.size()    #!20220102 
        #?print("down_conv3; "+ str(x3.size()))    #!20220102
        x = self.max_pool(x3)
        # print x.size()    #!20220102 
        #?print("max_pool; "+ str(x.size()))    #!20220102
        x = self.bottom_conv(x)

        # print x.size()    #!20220102 
        #?print("bottom_conv; "+ str(x.size()))    #!20220102
        x = self.up_conv1(x, x3)
        #print(x.size())
        # print x2.size()    #!20220102 
        #?print("up_conv1; "+ str(x.size()))    #!20220102
        x = self.up_conv2(x, x2)
        #?print("up_conv2; "+ str(x.size()))    #!20220102
        # print x.size()    #!20220102 
        x = self.up_conv3(x, x1)
        #?print("up_conv3; "+ str(x.size()))    #!20220102

        x = self.out_conv(x)
        #?print("out_conv; "+ str(x.size()))    #!20220102




        return x


        pass
#%%
if __name__ == '__main__':
    net = UNet()

    x = torch.rand((3,2,40))

    print(net(x).size())
# %%
