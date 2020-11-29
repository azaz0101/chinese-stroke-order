import torch
import torch.nn as nn
import numpy as np
import os
import time
import cv2
from PIL import Image
import torchsnooper
from torchvision.transforms import ToTensor , ToPILImage , ToPILImage
from lmodel import Local
from gmodel import Global
from dfs import dfs
import torch
import os
from tqdm import tqdm
np.set_printoptions(threshold=20000,linewidth=900)
torch.set_printoptions(threshold=20000,linewidth=900)
Lmodel = Local()
dir_path = 'train_data'
files = os.listdir(dir_path)
epoch = 10
use_cuda = 1
gpu_id = 1
if use_cuda:  
    torch.cuda.set_device(gpu_id)  
    Lmodel = Lmodel.cuda()

lists = []
for cnt,file in enumerate(files):
    f = open(dir_path+'/'+file)
    
    lists.append(f.readlines())
optimizer=torch.optim.Adam(Lmodel.parameters() ,lr=0.0001)
for eph in  range(1, epoch+1):
    for lis in enumerate(lists): 
        train_loss = 0.  
        to_tensor = ToTensor()
        to_pil = ToPILImage()
        celoss = nn.CrossEntropyLoss()
        bceloss = nn.BCELoss()
        img = to_tensor(np.zeros((109 , 109 , 1))).float()
        v = to_tensor(np.zeros((109 , 109 , 1))).float()
        le = to_tensor(np.zeros((109 , 109 ,1))).float()
        ls = to_tensor(np.zeros((109 , 109 , 1))).float()
        imgin = np.zeros((109,109))
        glo = 1
        arr = np.zeros(4,dtype = int)
        cnt = 0
        glo_cnt = 0
        end_pot = 0
        for i in range(len(lis[1])):
            
            x = lis[1][i].replace('\n','').split(',')
            x = list(map(int,x))
           
            img[0, x[1],x[0]] = 1
            imgin[x[1],x[0]] = 255
            if(x[3] == 1):
                glo = 1
                end_pot +=1
            if (x[3] == 0) & (glo == 1):
                glo = 0
                glo_cnt+=1
            cnt+=1
            arr = np.vstack([arr,x])
        torch_tar = torch.from_numpy(arr[1:])
        loc_cnt = cnt - glo_cnt - end_pot
        print(loc_cnt,glo_cnt,cnt)
        id = 0
        uv = img
        p = 0
        nx = 0
        ny = 0
        while(1):
        #break situation
            if(cnt-p<5):
                break
            #print('total' , cnt)
            #print('current' , p)
            target = torch.LongTensor(1)
            target = target.cuda()
            target[0] = torch_tar[p,0] % 109 + torch_tar[p,1] *109
            locate_x = target[0]/109
            locate_y = target[0]%109
            v[0 , locate_x , locate_y ] = 1
            uv[0 , locate_x , locate_y] = 0
            #clear ls and le 
            img_c = torch.zeros((109 , 109))
            seen = []         
            connected = dfs(seen , imgin , img_c , locate_x , locate_y)
            img_c = torch.unsqueeze(img_c ,0).float()
            head = (locate_x , locate_y)
            x = torch.cat((v , uv , img_c) , 0)
            x = torch.unsqueeze(x , 0)
            x = x.cuda()

            touched = 0
                #clear ls and le 
            le = to_tensor(np.zeros((109 , 109 ,1))).float()
            ls = to_tensor(np.zeros((109 , 109 , 1))).float()
            p+=1
            label = 1
            loc_touch = torch.zeros(1).cuda()
            
            loc_shift = torch.zeros([1,25]).cuda().requires_grad_()
          
            loc_touch_target = torch.zeros(1).cuda()
           
            loc_shift_target = torch.cuda.FloatTensor([1])
            
            #loc_touch = torch.zeros(1).requires_grad_()
            
            #loc_shift = torch.zeros([1,25]).requires_grad_()
          
            #loc_touch_target = torch.zeros(1)
           
            #loc_shift_target = torch.FloatTensor([1])
            optimizer.zero_grad()
            
            while(touched < 0.5):
                
                try:
                    touched , shifted = Lmodel(x , head)
                    touched = touched.cuda()
                    shifted = shifted.cuda()
                   
                except:
                    print("error")
                    break
               
                shift_target = torch.cuda.FloatTensor([1])
        
                shift_target[0] = torch.abs(torch_tar[p,0] - torch_tar[p-1,0]+2)%5 + torch.abs(torch_tar[p,1] - torch_tar[p-1,1]+2) * 5
                touch_target = torch.FloatTensor([1]).cuda()
            
                touch_target[0] = torch_tar[p,3]
                #print('torch_tar',torch_tar[p,3])
                #print(touched)
                touched = touched.squeeze(0)
                
                loc_touch = torch.cat([loc_touch,touched],dim=0)
                loc_touch_target = torch.cat([loc_touch_target,touch_target],dim = 0)
                #print(loc_touch_target.data)
                loc_shift = torch.cat([loc_shift,shifted],dim=0)
                loc_shift_target = torch.cat([loc_shift_target,shift_target],dim = 0)
                #print(loc_touch_target.size())
                shifted = torch.max(shifted , 1)[1]
                shifted_x = (shifted)/5-2
                shifted_y = (shifted)%5-2
                #avoid infinity loop               
                if((shifted_x == 0 and shifted_y == 0) or touch_target ==1):    
                    #print("here")
                    break
                nx = head[0]+shifted_x
                ny = head[1]+shifted_y
               # print(nx,ny)
                v[0,nx,ny] = 1
                uv[0,nx,ny] = 0
                head = (nx , ny)
         
                ls[0,nx,ny] = 1
                x = torch.cat((v , uv , img_c) , 0)
                x = torch.unsqueeze(x , 0)
                x = x.cuda()
                p+=1
                
            while(1):
                if(torch_tar[p,3]==1):
                    le[0,nx,ny] = 1
                    break
                else:
                    p+=1
            p+=1
            id+=1
           
            loc_shift = loc_shift[1:] 
            loc_shift_target = loc_shift_target[1:]
            
            loc_shift_target=torch.tensor(loc_shift_target, dtype=torch.long).cuda()

            
            
            loc_touch = loc_touch[torch.arange(loc_touch.size(0))!=0] 
            loc_touch_target = loc_touch_target[torch.arange(loc_touch_target.size(0))!=0] 
    
            #print('loc_touch_tar:',loc_touch_target.data)
            #print('touch:',loc_touch.data)
            
            touch_loss = bceloss(loc_touch,loc_touch_target)
            shift_loss = celoss(loc_shift,loc_shift_target)
     
            local_loss = touch_loss + shift_loss
            train_loss += local_loss
            #print(touch_loss,shift_loss)
            #print(local_loss)
            local_loss.backward()
            optimizer.step()

        print("eph:",eph,"loss:",train_loss)
        cout = to_pil(v.cpu().clone())
        cout.save(f'./loc_out_{eph}.jpg')
        if(lis[0]%1000==0): 
            torch.save(Lmodel.state_dict(),f'./pretrained/local_train_{eph}_{lis[0]}.pt')
        #if(eph % 10 ==0):
        
        print('img:',lis[0],'loss:',train_loss)
        #optimizer.step()
        
