from time import sleep
import cv2
from algorithm_1 import *
from pygame import mixer
status=0
mixer.music.load("D:\Projects\Trinetra\WhatsApp Ptt 2022-11-21 at 10.49.24 AM-[AudioTrimmer.com].mp3")

cap=cv2.VideoCapture("y2mate.com - Feature Space Optimization for Semantic Video Segmentation  CityScapes Demo 02_1080p.mp4")

while cap.isOpened():
    command=None
    ret,frame=cap.read()
    frame=cv2.resize(frame,(720,360))
    df=path_find(frame)
    
    angle=list(df['Angle'])
    if any(abs(ang)>=85 for ang in angle):
        command='stop'
        if status==0:
            print("Playing Music")
            mixer.music.play()

        if mixer.music.get_busy():
            status=1
            print('busy')
        else:
            status=0
            mixer.stop()
            print("Command Completed")

    elif abs(angle[0])>=0 and abs(angle[0])<=8:
        command='straight'
    
    elif abs(angle[0])>8 and abs(angle[0])<85:
        command='turning'
    
    print(command)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()