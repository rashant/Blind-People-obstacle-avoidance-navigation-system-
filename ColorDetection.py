import time

import cv2
import numpy as np


def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("HUE Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255 ,empty)
cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)


cap=cv2.VideoCapture("y2mate.com - Feature Space Optimization for Semantic Video Segmentation  CityScapes Demo 02_1080p.mp4")
counter=0
while cap.isOpened():
    counter+=1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT)==counter:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        counter=0
    s,img=cap.read()
    img=cv2.resize(img,(400,256))
    imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hmin=cv2.getTrackbarPos("HUE Min","HSV")
    hmax=cv2.getTrackbarPos("HUE Max","HSV")
    smin=cv2.getTrackbarPos("SAT Min","HSV")
    smax=cv2.getTrackbarPos("SAT Max","HSV")
    vmin=cv2.getTrackbarPos("VALUE Min","HSV")
    vmax=cv2.getTrackbarPos("VALUE Max","HSV")

    lower=np.array([hmin,smin,vmin])
    upper=np.array([hmax,smax,vmax])
    print(lower,upper)
    mask=cv2.inRange(imgHsv,lower,upper)
    result=cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("Original",img)
    cv2.imshow("mask",mask)
    cv2.imshow("result",result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(mask.shape)

cv2.destroyAllWindows()


# while True:
#     success,img=cap.read()
#     img=cv2.flip(img,1)
#     img = cv2.resize(img, (400, 256))
#     imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#     hmin=cv2.getTrackbarPos("HUE Min","HSV")
#     hmax=cv2.getTrackbarPos("HUE Max","HSV")
#     smin=cv2.getTrackbarPos("SAT Min","HSV")
#     smax=cv2.getTrackbarPos("SAT Max","HSV")
#     vmin=cv2.getTrackbarPos("VALUE Min","HSV")
#     vmax=cv2.getTrackbarPos("VALUE Max","HSV")

#     lower=np.array([hmin,smin,vmin])
#     upper=np.array([hmax,smax,vmax])
#     print(lower,upper)
#     mask=cv2.inRange(imgHsv,lower,upper)
#     result=cv2.bitwise_and(img,img,mask=mask)

#     cv2.imshow("Original",img)
#     cv2.imshow("mask",mask)
#     cv2.imshow("result",result)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
  
# # After the loop release the cap object
# cap.release()
# # Destroy all the windows
# cv2.destroyAllWindows()