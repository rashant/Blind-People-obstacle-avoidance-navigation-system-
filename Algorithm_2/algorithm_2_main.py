from algorithm_2 import *
import cv2
import time

start_time=time.time()
cap=cv2.VideoCapture(r"y2mate.com - Feature Space Optimization for Semantic Video Segmentation  CityScapes Demo 02_1080p.mp4")
try:    
    while cap.isOpened():
        ret,frame=cap.read()
        frame=cv2.resize(frame,(360,360))
        grid_img,mask,grid_map=gridMap(frame)

        cv2.imshow("Image",frame)
        cv2.imshow("Grid Image",grid_img)
        cv2.imshow("Mask Image",mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except:
    print()

print(time.time()-start_time)