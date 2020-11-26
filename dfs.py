def dfs(seen , img , img_connected , x , y):
    for i in [-2 , -1 , 0 , 1 , 2]:
        for j in [-2 , -1 , 0 , 1 , 2]:
            if (i,j)!=(0,0):
                nx = x + i
                ny = y + j
                if(not (nx>0 and nx<109 and ny>0 and ny<109)):
                    continue
                if(img[nx][ny]!=0 and (nx,ny) not in seen):
                    seen.append((nx , ny))
                    dfs(seen , img , img_connected , nx , ny)
                    img_connected[nx,ny] = 1
