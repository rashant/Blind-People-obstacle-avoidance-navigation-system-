import cv2
import numpy as np
import imutils
import pandas as pd
from SpeechCommand import *

'''Angle finding function'''
def findAngle(x1,y1,x2,y2,x3,y3):
    radian=np.arctan2(y2-y1,x2-x1) - np.arctan2(y3-y1,x3-x1)
    #angle=round(np.abs(radian*180/np.pi),0)
    angle=round(radian*180/np.pi,0)
    return angle

def path_find(img):
    area_list=[]
    segment_list=[]
    x1_list=[]
    y1_list=[]
    x2_list=[]
    y2_list=[]
    x3_list=[]
    y3_list=[]
    angle_list=[]
    segment=1
    rows,cols,_=img.shape

    prevx,prevy=int(cols/2),cols

    # lower=np.array([146,  37,  93])
    # upper=np.array([157, 107, 150])
    lower=np.array([0,  0,  129])
    upper=np.array([179, 115, 255])
    imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    line_frame=img.copy()
    mask=cv2.inRange(imgHsv,lower,upper)
    area_list=[]
    segment_list=[]
    x1_list=[]
    y1_list=[]
    x2_list=[]
    y2_list=[]
    x3_list=[]
    y3_list=[]
    angle_list=[]
    segment=1

    '''FINDING NUMBER OF WHITE PIXEL IN EACH IMAGE'''
    counter=0
    for i in range(255,0,-40):
        crop_img=mask[i-40:i,0:cols]
        cnts=cv2.findContours(crop_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts=imutils.grab_contours(cnts)

        for c in cnts:
            area=cv2.contourArea(c)
            if(area>2000):
                segment_list.append(segment)
                segment+=1
                cv2.drawContours(crop_img,[c],-1,(0,0,139),1)
                
                M=cv2.moments(c)
                cx=int(M["m10"]/M["m00"])
                cy=int(M["m01"]/M["m00"])

                x1,y1=prevx,prevy
                x2,y2=prevx,cy+i
                x3,y3=cx,cy+i

                x1_list.append(x1)
                y1_list.append(y1)

                x2_list.append(x2)
                y2_list.append(y2)

                x3_list.append(x3)
                y3_list.append(y3)

                angle=findAngle(x1,y1,x2,y2,x3,y3)
                angle_list.append(angle)

                area_list.append(area)

                if abs(angle)>80:
                    cv2.putText(img,"Stop",(90,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1,cv2.LINE_AA)
                    
                    #cv2.putText(img,"Played",(90,100), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1,cv2.LINE_AA)
                    #command_play()

                cv2.line(img, (prevx,prevy), (prevx,cy+i), (0, 255, 0), 1)
                cv2.line(img, (prevx,prevy), (cx,cy+i), (0, 255, 0), 1)
                cv2.line(line_frame, (prevx,prevy), (cx,cy+i), (0, 255, 0), 1)
                
                cv2.putText(img,f"({cx},{cy+i})",(cx+3,cy+i), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
                #cv2.putText(line_frame,f"({int(cols/2)},{cols})",(int(cols/2)+5,cols), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
                cv2.putText(img,f"Angle:- {angle}",(cx+3,cy+i+15), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
                
                cv2.circle(img,(cx,cy+i),2,(3,140,252),-1)
                cv2.circle(line_frame,(cx,cy+i),2,(3,140,252),-1)
                cv2.circle(img,(int(cols/2),cols),5,(0,255,0),-1)

                xmask=cv2.resize(mask,(720,360))
                ximg=cv2.resize(img,(720,360))
                xline_frame=cv2.resize(line_frame,(720,360))
                xcrop_img=cv2.resize(crop_img,(720,360))

                xmask=cv2.cvtColor(xmask,cv2.COLOR_GRAY2BGR)
                xcrop_img=cv2.cvtColor(xcrop_img,cv2.COLOR_GRAY2BGR)

                hor=np.concatenate((ximg,xmask),axis=0)
                vert=np.concatenate((xline_frame,xcrop_img),axis=0)
                final=np.concatenate((hor,vert),axis=1)
                cv2.imshow("Path",final)
                prevx,prevy=cx,cy+i
                counter+=1
        if counter==3:
            break
        
    dic={"Segment":segment_list,"X1":x1_list,"X2":x2_list,"X3":x3_list,"Y1":y1_list,"Y2":y2_list,"Y3":y3_list,"Area":area_list,"Angle":angle_list}
    df=pd.DataFrame(dic)
    return df