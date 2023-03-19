from collections import Counter
from path_finding import *
import cv2
import time
from algorithm_2 import *
import numpy as np

cap=cv2.VideoCapture("y2mate.com - Feature Space Optimization for Semantic Video Segmentation  CityScapes Demo 02_1080p.mp4")
fps_list=[]
prev_frame_time=0
new_frame_time=0
font=cv2.FONT_HERSHEY_COMPLEX

while(cap.isOpened()):
    ret,frame=cap.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(360,360))
    grid_img,mask,grid_map,path_map=gridMap(frame)

    # grid_map=np.array(grid_map)
    # rows,columns=grid_map.shape
    # di=[1,1,1]
    # dj=[0,1,-1]
    # vis=np.zeros(shape=(9,18))
    # ans=[]
    # move=""
    # final_path=path(i=rows-1,j=int(columns/2),matrix=grid_map,m=rows,n=columns,ans=ans,move=move,vis=vis,di=di,dj=dj)
    # print(final_path)
    new_frame_time=time.time()

    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    fps=str(int(fps))
    fps_list.append(int(fps))
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.imshow("Grid Image",grid_img)
    cv2.imshow("Mask Image",mask)
    cv2.imshow("Map Image",path_map)
    print("\n\n")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

print(sum(fps_list)/len(fps_list))
