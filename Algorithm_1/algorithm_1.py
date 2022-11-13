import cv2
import numpy as np
import imutils
import statistics
import pandas as pd

img_number=88
img=cv2.imread(f"D:\Projects\Trinetra\Algorithm_1\image_{img_number}.jpg")
rows,cols,_=img.shape

print("Rows:- ",rows)
print("Columns:- ",cols)

prevx,prevy=int(cols/2),cols

#crop_img=img[0:32,0:256]

'''SPLITTING THE IMAGES'''
# for i in range(0,256,8):
#     crop_img=img[i:i+8,0:256]

#     cv2.imshow("Image",img)
#     cv2.imshow("Cropped Image",crop_img)
#     cv2.waitKey(0)

'''Angle finding function'''
def findAngle(x1,y1,x2,y2,x3,y3):
    radian=np.arctan2(y2-y1,x2-x1) - np.arctan2(y3-y1,x3-x1)
    angle=round(np.abs(radian*180/np.pi),0)
    return angle

print("\n\nAlgorithm\n")

lower=np.array([136, 111,  86])
upper=np.array([156, 148, 144])
imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

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
for i in range(255,0,-8):
    crop_img=mask[i-8:i,0:256]
    cnts=cv2.findContours(crop_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    for c in cnts:
        area=cv2.contourArea(c)
        if(area>1000):
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


            # print(f"Point 1:- {prevx,prevy}")
            # print(f"Point 2:- {prevx,cy+i}")
            # print(f"Point 3:- {cx,cy+i}\n")
            
            cv2.line(img, (prevx,prevy), (prevx,cy+i), (0, 255, 0), 1)
            cv2.line(img, (prevx,prevy), (cx,cy+i), (0, 255, 0), 1)
            
            cv2.putText(img,f"({cx},{cy+i})",(cx+3,cy+i), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
            cv2.putText(img,f"({int(cols/2)},{cols})",(int(cols/2)+5,cols), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
            cv2.putText(img,f"Angle:- {angle}",(50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
            
            cv2.circle(img,(cx,cy+i),1,(3,140,252),-1)
            cv2.circle(img,(int(cols/2),cols),5,(0,255,0),-1)

            xmask=cv2.resize(mask,(720,360))
            ximg=cv2.resize(img,(720,360))

            cv2.imshow("mask",xmask)
            cv2.imshow("image",ximg)
            cv2.imshow("crop",crop_img)
            img=cv2.imread(f"D:\Projects\Trinetra\Algorithm_1\image_{img_number}.jpg")

            prevx,prevy=cx,cy+i
            counter+=1

    if counter==7:
        break
    cv2.waitKey(0)


dic={"Segment":segment_list,"X1":x1_list,"X2":x2_list,"X3":x3_list,"Y1":y1_list,"Y2":y2_list,"Y3":y3_list,"Area":area_list,"Angle":angle_list}
df=pd.DataFrame(dic)
print(df)
print(f"Total walkable area:- {round(sum(area_list),2)}")
print(f"Average walkable area of each segment:- {round(statistics.mean(area_list),2)}")