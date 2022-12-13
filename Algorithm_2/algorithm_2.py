import cv2
import numpy as np
import time

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

    grid_map=np.zeros([10,18],dtype=float)


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
        
    return grid_img,mask,grid_map
    