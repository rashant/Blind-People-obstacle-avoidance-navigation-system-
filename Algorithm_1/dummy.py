import cv2
import numpy as np
import imutils
import statistics
import matplotlib.pyplot as plt
img=cv2.imread("D:\Projects\Trinetra\Algorithm_1\image_28.jpg")

rows,cols,_=img.shape

print("Rows:- ",rows)
print("Columns:- ",cols)
print(img.shape)
points_list=[]
#crop_img=img[0:32,0:256]

'''SPLITTING THE IMAGES'''
# for i in range(0,256,8):
#     crop_img=img[i:i+8,0:256]

#     cv2.imshow("Image",img)
#     cv2.imshow("Cropped Image",crop_img)
#     cv2.waitKey(0)

prevx,prevy=int(cols/2),int(cols/2)

print("\n\nAlgorithm\n")

lower=np.array([136, 111,  86])
upper=np.array([156, 148, 144])
imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask=cv2.inRange(imgHsv,lower,upper)
area_list=[]
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
            print("Segment - {}".format(segment))
            segment+=1
            
            cv2.drawContours(crop_img,[c],-1,(0,0,139),1)
            area_list.append(area)
            print("Area:- ",area)
            M=cv2.moments(c)
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])
            points_list.append([cx,cy+i])
            print(points_list)
            
            print("Co-ordinates:- {}\n".format((cx,cy)))
            if(cx<int(cols/2)-5):
                cv2.putText(img,f"Left",(50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
            elif(cx>int(cols/2)+5):
                cv2.putText(img,f"Right",(50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
            elif(cx>=int(cols/2)-5 and cx<=int(cols/2)-5):
                cv2.putText(img,f"Straight",(50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
            cv2.putText(img,f"({cx},{cy+i})",(cx+3,cy+i), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(img,(cx,cy+i),1,(3,140,252),-1)
            cv2.circle(img,(int(cols/2),cols),4,(0,255,0),1)
            cv2.line(img, (prevx,prevy+i), (prevx,0), (0,255,0), 1)
            cv2.line(img, (prevx+5,prevy+i), (prevx+5,0), (0,255,0), 1)
            cv2.line(img, (prevx-5,prevy+i), (prevx-5,0), (0,255,0), 1)
            xmask=cv2.resize(mask,(360,360))
            ximg=cv2.resize(img,(360,360))
            cv2.imshow("mask",xmask)
            cv2.imshow("image",ximg)
            cv2.imshow("crop",crop_img)
            prevx,prevy=cx,cy
            counter+=1
            img=cv2.imread("D:\Projects\Trinetra\Algorithm_1\image_28.jpg")
    if counter==5:
        break
    cv2.waitKey(0)
print(points_list)
print(f"Total walkable area:- {round(sum(area_list),2)}")
print(f"Average walkable area of each segment:- {round(statistics.mean(area_list),2)}")