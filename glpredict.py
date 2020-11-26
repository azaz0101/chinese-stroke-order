import torch
import torch.nn as nn
import numpy as np
import os
import time
import cv2
from PIL import Image
from torchvision.transforms import ToTensor , ToPILImage , ToPILImage
from lmodel import Local
from gmodel import Global
from dfs import dfs
import sys

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=200)

Lpretrained = './pretrained/local_train9000.pt'
Lmodel = Local()
Lmodel.load_state_dict(torch.load(Lpretrained , map_location=torch.device('cpu')))

Gpretrained = './pretrained/glomodel_100.pt'
Gmodel = Global()
Gmodel.load_state_dict(torch.load(Gpretrained , map_location=torch.device('cpu')))


#np.set_printoptions(threshold = np.inf)


if __name__ == '__main__':

    path = sys.argv[1]
    celoss = nn.CrossEntropyLoss()
    bceloss = nn.BCELoss()


    to_tensor = ToTensor()
    to_pil = ToPILImage()
    imgin = np.array(Image.open(path))
    imgin = np.where(imgin<100 , 0 , 255)
    img = to_tensor(Image.open(path)).float()
    v = to_tensor(np.zeros((109 , 109 , 1))).float()
    le = to_tensor(np.zeros((109 , 109 ,1))).float()
    ls = to_tensor(np.zeros((109 , 109 , 1))).float()
    uv = img
    id = 1
    image_list = list() 
    #total point 
    cnt = 0
    for i in range(109):
        for j in range(109):
            if(imgin[i][j]==255):
                cnt += 1  
    p = 0
    c = 0
    ans = ''
    while(1):
        #break situation
        p = 0
        #p = c
        for i in range(109):
            for j in range(109):
                if v[0,i,j]>0.5:
                    p += 1
        if(cnt-p<5):
            break
        print('total' , cnt)
        print('current' , p)

        gx = torch.cat((v , uv , le , ls) , 0)
        gx = torch.unsqueeze(gx , 0)
        #Global model
        gx = gx
        locate = Gmodel(gx) 
 
 
        locate = torch.max(locate.view(locate.shape[0] , -1) , 1)[1]
        locate_x = locate/109
        locate_y = locate%109
        input = torch.cat((locate_y,locate_x),0)
        
        #print('locate:',locate_x,locate_y)
        #print(torch_tar[p])

        #Local model 
        v[0 , locate_x , locate_y ] = 1
        uv[0 , locate_x , locate_y] = 0
        img_c = torch.zeros((109 , 109))
        seen = []
        connected = dfs(seen , imgin , img_c , locate_x , locate_y)
        img_c = torch.unsqueeze(img_c ,0).float()
        head = (locate_x , locate_y)
        x = torch.cat((v , uv , img_c) , 0)
        image = to_pil(img_c)
        x = torch.unsqueeze(x , 0)
        touched = 0


        #clear ls and le 
        le = to_tensor(np.zeros((109 , 109 ,1))).float()
        ls = to_tensor(np.zeros((109 , 109 , 1))).float()
        c = p
        while(touched < 0.5):
            #print("local:",c)
            try:
                touched , shifted = Lmodel(x , head)
                #touched = touched.cuda()
                #shifted = shifted.cuda()
                #print('touched:',touched)
            except:
                print("error")
                break
            shifted = torch.max(shifted , 1)[1]
            shifted_x = (shifted)/5-2
            shifted_y = (shifted)%5-2
            #print('shift:',shifted_x,shifted_y)
           #avoid infinity loop
            c+=1
            if shifted_x==0 and shifted_y==0:
                break
            nx = head[0]+shifted_x
            ny = head[1]+shifted_y
            v[0,nx,ny] = 1
            uv[0,nx,ny] = 0
            head = (nx , ny)
            ls[0,nx,ny] = 1
            x = torch.cat((v , uv , img_c) , 0)
            x = torch.unsqueeze(x , 0)
            image_list.append(to_pil(v.cpu().clone()))
            #print(nx,ny)
            if (touched.item() < 0.5):
                t = [str(ny.item()), str(nx.item()), str(id), '0']
            else :
                t = [str(ny.item()), str(nx.item()), str(id), '1']
            #print(t)
            s = ','.join(t)
            s += '\n'
            ans += s
        #print('output:',nx,ny)
        #print("output:",touched)
        le[0,nx,ny] = 1
        image = to_pil(v.cpu().clone())
        #imageuv = to_pil(img_c.cpu().clone())
        tmp = path.split('/')[3]
        tmp = tmp.split('.')[0]
        image.save(f'./output_Img/{tmp}_{id}.jpg')
        #imageuv.save(f'./unvis/uv_{id}.jpg')
        id += 1
        first = False
    #image_list[0].save('./output/out.gif', save_all = True, append_images = image_list[1:])
    with open(f'./output_Stk/{tmp}.txt','w') as f:
        f.write(ans)
   
