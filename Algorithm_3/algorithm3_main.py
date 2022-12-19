# import the opencv library
from time import sleep
import cv2
from algorithm3 import *
from pygame import mixer

mixer.init()
# define a video capture object
#vid = cv2.VideoCapture(r"y2mate.com - Feature Space Optimization for Semantic Video Segmentation  CityScapes Demo 02_1080p.mp4")

vid = cv2.VideoCapture(0)

command_list=[-1]

while(True):
    ret, frame = vid.read()
    frame=cv2.resize(frame,(360,360))

    frame=cv2.flip(frame,1)

    #grid_img,mask,grid_map,path_map,directions=path(frame)
    grid_img,mask,directions=path(frame)

    direction=max(directions)
    frame = cv2.putText(frame, str(direction), (1, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

    try:
        match direction:
            case 0:
                if direction not in command_list:
                    command_list.pop()
                    command_list.append(direction)
                    mixer.music.load("commands/straight.mp3")
                    mixer.music.play()
                    print('S')     

                if mixer.music.get_busy():
                    pass
                else:
                    mixer.stop()
            
            case 1:
                if direction not in command_list:
                    command_list.pop()
                    command_list.append(direction)
                    mixer.music.load("commands/right.mp3")
                    mixer.music.play()
                    print('R')

                if mixer.music.get_busy():
                    pass
                else:
                    mixer.stop()
            
            case 2:
                if direction not in command_list:
                    command_list.pop()
                    command_list.append(direction)
                    mixer.music.load("commands/left.mp3")
                    mixer.music.play()
                    print('L')

                if mixer.music.get_busy():
                    pass
                else:
                    mixer.stop()
            
            case 3:
                if direction not in command_list:
                    command_list.pop()
                    command_list.append(direction)
                    mixer.music.load("commands/stop.mp3")
                    mixer.music.play()
                    print('ST')

                if mixer.music.get_busy():
                    pass
                else:
                    mixer.stop()
    except:
        pass

    
    grid=np.vstack((mask,grid_img))

    cv2.imshow('frame', frame)
    cv2.imshow('Grid', grid)

    # cv2.imshow("Grid Image",grid_img)
    # cv2.imshow("Mask Image",mask)
    # cv2.imshow("Map Image",path_map)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


# import cv2
# from algorithm3 import *

# #frame=cv2.imread(r'Algorithm_2\test.png')
# frame=cv2.imread(r'Algorithm_2\test2.png')
# frame=cv2.resize(frame,(360,360))
# print(frame.shape)
# grid_img,mask,grid_map,path_map=path(frame)

# cv2.imshow("frame",frame)
# cv2.imshow("Grid Image",grid_img)
# cv2.imshow("Mask Image",mask)
# cv2.imshow("Map Image",path_map)
# cv2.waitKey(0)