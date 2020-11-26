import torch.nn as nn
import torch
from torch.nn import functional as F

# refrence: https://github.com/chenyuntc/pytorch-book/blob/master/chapter04-neural_network/chapter4.ipynb

class ResidualBlock(nn.Module):
    def __init__(self,in_c,out_c,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.Lpath = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,1,1,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,1,1,bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.Rpath = shortcut
    def forward(self,x):
        out = self.Lpath(x)
        residual = x if self.Rpath is None else self.Rpath(x)
        out+=residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self , in_c):
        super(ResNet,self).__init__()
        # pre可以做更改-----------------------
        self.pre = nn.Sequential(
            nn.Conv2d(in_c,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # num block可以做更改-----------------
        self.layer1 = self.make_layer(16,16,3)
        self.layer2 = self.make_layer(16,16,4)
        self.layer3 = self.make_layer(16,32,6)
        self.layer4 = self.make_layer(32,64,3)

    def make_layer(self,in_c,out_c,block_num):
        # shortcut可以做更改------------------
        shortcut = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1,bias=False),
            nn.BatchNorm2d(out_c)
        )
        layers = []
        layers.append(ResidualBlock(in_c,out_c,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(out_c,out_c))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

if __name__ == '__main__':
    model = ResNet(in_c=3)
    img = torch.randn((5,3,100,100))
    out = model(img)
    print(out.shape)




    
