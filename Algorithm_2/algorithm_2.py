import cv2
import numpy as np
import time
from path_finding import *

def gridMap(frame):

    frame=frame[180:,:,:]
    img=frame
    lower=np.array([146,  37,  93])
    upper=np.array([157, 107, 150])

    imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHsv,lower,upper)

    black_block= np.zeros([20, 20], dtype = float)
    white_block= np.ones([20, 20], dtype = float)

    grid_img= np.zeros([180,360],dtype=float)

    grid_map=np.zeros([9,18],dtype=float)

    path_map=np.zeros([180,360],dtype=float)


    r_step=20
    c_step=20
    ini_r=0
    ini_c=0

    for i in range(9):
        for j in range(18):
            avg_list=[]
            sumx=0
            image_block=mask[ini_r:r_step,ini_c:c_step]

            for k in image_block:
                sumx+=(sum(k)/len(k))

            average=sumx/18
            avg_list.append(round(average,0))

            if round(average,0)>=210:
                grid_img[ini_r:r_step,ini_c:c_step]=white_block
                grid_map[i][j]=1
            else:
                grid_img[ini_r:r_step,ini_c:c_step]=black_block
                grid_map[i][j]=0

            ini_c=c_step
            c_step+=20

        ini_r=r_step
        r_step+=20
        ini_c=0
        c_step=20
    
    grid_map=np.array(grid_map)
    rows,columns=grid_map.shape
    di=[1,1,1]
    dj=[0,1,-1]
    vis=np.zeros(shape=(9,18))
    move=""
    final_path=path(i=rows-1,j=int(columns/2),matrix=grid_map,m=rows,n=columns,move=move,vis=vis,di=di,dj=dj)
    print(final_path)

    path_map[160:180,160:180]=white_block
    rc_step=20
    r_init=160
    c_init=160
    try:
        directions=final_path.strip().split(' ')
        for i in directions:
            match i:
                case 'S':
                    rl=r_init-rc_step
                    rr=r_init
                    cl=c_init
                    cr=c_init+rc_step
                    path_map[rl:rr,cl:cr]=white_block
                    r_init=rl

                case 'DL':
                    rl=r_init-rc_step
                    rr=r_init
                    cl=c_init-rc_step
                    cr=c_init
                    path_map[rl:rr,cl:cr]=white_block
                    r_init=rl
                    c_init=cl

                case 'DR':
                    rl=r_init-rc_step
                    rr=r_init
                    cl=c_init+rc_step
                    cr=c_init+rc_step*2
                    path_map[rl:rr,cl:cr]=white_block
                    r_init=rl
                    c_init=cl
    except:
        pass
    sleep(1)
    return grid_img,mask,grid_map,path_map
    