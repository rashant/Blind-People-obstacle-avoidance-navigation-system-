from time import sleep
import cv2
from algorithm_1 import *
from pygame import mixer
import pandas as pd

com_list=[]

mixer.init()
cap=cv2.VideoCapture("D:\Projects\Trinetra\y2mate.com - Feature Space Optimization for Semantic Video Segmentation  CityScapes Demo 02_1080p.mp4")
# cap=cv2.VideoCapture(1)

command_list=['No']

#RIGHT---> -VE 
#LEFT----> +VE

while cap.isOpened():
    command=None
    ret,frame=cap.read()
    frame=cv2.resize(frame,(720,360))
    df=path_find(frame)
    
    angle=list(df['Angle'])
    try:
        if any(abs(ang)>=85 for ang in angle):
            command='stop'
            if command not in command_list:
                command_list.pop()
                command_list.append(command)
                mixer.music.load("commands/stop.mp3")
                mixer.music.play()
                print('command played')

            if mixer.music.get_busy():
                pass
            else:
                
                mixer.stop()
        
        elif abs(angle[0])>=0 and abs(angle[0])<=8:
            command='straight'
            if command not in command_list:
                command_list.pop()
                command_list.append(command)
                mixer.music.load("commands/straight.mp3")
                mixer.music.play()
                print('command played')

            if mixer.music.get_busy():
                pass
            else:
                mixer.stop()
        
        elif abs(angle[-1])>8 and abs(angle[-1])<85:

            if angle[-1]<0:
                command='right'
                if command not in command_list:
                    command_list.pop()
                    command_list.append(command)
                    mixer.music.load("commands/right.mp3")
                    mixer.music.play()
                    print('command played')

                if mixer.music.get_busy():
                    pass
                else:
                    mixer.stop()
        
            else:
                command='left'
                if command not in command_list:
                    command_list.pop()
                    command_list.append(command)
                    mixer.music.load("commands/left.mp3")
                    mixer.music.play()
                    print('command played')

                if mixer.music.get_busy():
                    pass
        
                else:
                    mixer.stop()



    except:
        pass
    
    
    print(command_list,command)
    com_list.append(command)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df=pd.DataFrame({'Commands':com_list})
df.to_csv('commands.csv')