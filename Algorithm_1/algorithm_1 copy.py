from time import sleep
import cv2
from algorithm_1 import *

cap=cv2.VideoCapture("y2mate.com - Feature Space Optimization for Semantic Video Segmentation  CityScapes Demo 02_1080p.mp4")
while cap.isOpened():
    ret,frame=cap.read()
    frame=cv2.resize(frame,(720,360))
    df=path_find(frame)
    print(df)
    #sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()