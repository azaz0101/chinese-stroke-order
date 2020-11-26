import torch
import torch.nn as nn
import Resnet

class Local(nn.Module):
    
    def __init__(self):
        super(Local,self).__init__()
        self.resnet = Resnet.ResNet(in_c=3) 
        self.fc_touched = nn.Sequential(
            nn.Linear(1600 , 800),
            nn.ReLU(True),
            nn.Linear(800 , 800),
            nn.ReLU(True),
            nn.Linear(800 , 1),
            nn.Sigmoid()
        )
        self.fc_shifted = nn.Sequential(
            nn.Linear(1600 , 800),
            nn.ReLU(True),
            nn.Linear(800 , 800),
            nn.ReLU(True),
            nn.Linear(800 , 25)
        )
    
    def forward(self,x,central):
        x = self.resnet(x)    
        #residual
        #print(central)
        if(central[0]-2>=0 and central[0]+3<=109 and central[1]-2>=0 and central[1]+3<=109):
            x2 = x[:,:,central[0]-2:central[0]+3,central[1]-2:central[1]+3]
            x3 = x2.contiguous().view(x.shape[0] , -1)
        else:
            return None
        touched = self.fc_touched(x3)
        shifted = self.fc_shifted(x3)
        
        return touched , shifted

#-----------------------------------


if __name__ == '__main__':

    device = torch.device('cuda:0')
    img = torch.randn((5 , 3 , 109 ,109))
    img = img.to(device)
    model = Local()
    model = model.to(device)
    touched , shifted = model(img , (10 , 10))
    #gt
    t = torch.randn((5 , 1))
    s = torch.randn((5 , 25))
    t , s = t.to(device , dtype = torch.float) , s.to(device , dtype = torch.long)
    
    bceloss = nn.BCELoss()
    celoss = nn.CrossEntropyLoss()
    print(s)
    loss = bceloss(touched , t) + celoss(shifted , torch.max(s , 1)[1])
    print(loss)


    '''
    #test loss function
    loss = nn.MSELoss()
    #test caculation
    target = model(img,(10 , 10))
    #test caculation of gredient
    mse = loss(label , target)
    #test back propagation
    mse.backward()
    '''
    
        
    
