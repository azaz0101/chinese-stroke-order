import torch
import torch.nn as nn
import Resnet

class Global(nn.Module):
    def __init__(self):
        super(Global,self).__init__()
        self.resnet = Resnet.ResNet(in_c=4)
        self.conv1 = nn.Conv2d(64,1,1,1,bias=False) 
        #self.softmax = nn.Softmax()  
    def forward(self,x):
        x = self.resnet(x)
        x = self.conv1(x)
        x = x.view(x.shape[0] , -1)
        return x
