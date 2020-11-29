# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os
import time
from PIL import Image
from torchvision.transforms import ToTensor , ToPILImage , ToPILImage
from lmodel import Local
from gmodel import Global
from dfs import dfs
import torch
import os
from tqdm import tqdm
dir_path = 'train_data'
files = os.listdir(dir_path)
epoch = 100
Gmodel = Global()
gpu_id = 1
if use_cuda:  
    torch.cuda.set_device(gpu_id)  
    Gmodel = Gmodel.cuda()

optimizer=torch.optim.Adam(Gmodel.parameters() ,lr=0.0001)
for eph in  range(1, epoch+1):
    train_loss = 0.
    for i,file in tqdm(enumerate(files)):    
        f = open(dir_path+'/'+file)

        to_tensor = ToTensor()
        to_pil = ToPILImage()
        celoss = nn.CrossEntropyLoss()
        img = to_tensor(np.zeros((109 , 109 , 1))).float()
        v = to_tensor(np.zeros((109 , 109 , 1))).float()
        le = to_tensor(np.zeros((109 , 109 ,1))).float()
        ls = to_tensor(np.zeros((109 , 109 , 1))).float()

        glo = 1
        arr = np.zeros(4,dtype = int)
        cnt = 0
        glo_cnt = 0
        for line in f:
            x = line.replace('\n','').split(',')
            for i in range(0,len(x)):
                x[i] = int(x[i])
            img[0, x[1],x[0]] = 1
            if(x[3] == 1):
                glo = 1
            if (x[3] == 0) & (glo == 1):
                glo = 0
                glo_cnt+=1
            cnt+=1
            arr = np.vstack([arr,x])
        torch_tar = torch.from_numpy(arr[1:])

        uv = img
        p = 0
	glo_loc = torch.zeros([1,11881]).cuda()
	glo_target = torch.cuda.LongTensor([1])
	optimizer.zero_grad()
        while(1):
        #break situation
            if(p+1 == cnt):
                break
            gx = torch.cat((v , uv , le , ls) , 0)
            gx = torch.unsqueeze(gx , 0)
        #Global model
            gx = gx
            gx = gx.cuda()
            locate = Gmodel(gx)
            locate = locate.cuda()
            target = torch.LongTensor([1])
            target = target.cuda()
            target[0] = torch_tar[p,0] % 109 + torch_tar[p,1] *109
           
	    glo_loc = torch.cat([glo_loc,locate],dim=0)
  	    glo_target = torch.cat([glo_target,target],dim=0)

            #loss = celoss(locate,target)
            #train_loss+=loss
            #loss.backward()
            locate = torch.max(locate.view(locate.shape[0] , -1) , 1)[1]
            locate_x = locate/109
            locate_y = locate%109
            v[0 , locate_x , locate_y ] = 1
            uv[0 , locate_x , locate_y] = 0
            #clear ls and le 
            le = to_tensor(np.zeros((109 , 109 ,1))).float()
            ls = to_tensor(np.zeros((109 , 109 , 1))).float()
            p+=1
            
            while(torch_tar[p,3] < 1):
                ny = torch_tar[p,0]
                nx = torch_tar[p,1]
                v[0, nx, ny] = 1
                uv[0, nx, ny] = 0
                ls[0,nx,ny] = 1
                p+=1
            #print("output")
            le[0,nx,ny] = 1
	glo_loc = glo_loc[1:]
	glo_target = glo_target[1:]
	loss = celoss(glo_loc,glo_target)
        train_loss+=loss
        loss.backward()
	optimizer.step()
    
	#print('file_loss:',loss)
    print('epoch:',eph,'loss:',train_loss / len(files))
    #if(eph % 10==0):
    torch.save(Gmodel.state_dict(),'./pretrained/glomodel_{eph}.pt') 
