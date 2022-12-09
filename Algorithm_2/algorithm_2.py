import cv2
import numpy as np
import time

code_start_time = time.time()

img=cv2.imread('Algorithm_2\image1.jpg')

img=cv2.resize(img,(1024,720))
rows=720
columns=1024

lower=np.array([146,  37,  93])
upper=np.array([157, 107, 150])

imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(imgHsv,lower,upper)

black_block= np.zeros([60, 85], dtype = float)
white_block= np.ones([60, 85], dtype = float)

grid_img= np.zeros([720,1024],dtype=float)

grid_map=np.zeros([12,12],dtype=float)


r_step=60
c_step=85
ini_r=0
ini_c=0


loop_start_time = time.time()
for i in range(12):
    for j in range(12):
        avg_list=[]
        sumx=0
        image_block=mask[ini_r:r_step,ini_c:c_step]

        for k in image_block:
            sumx+=(sum(k)/len(k))

        average=sumx/12



        if round(average,0)>=130:
            grid_img[ini_r:r_step,ini_c:c_step]=white_block
            grid_map[i][j]=1
        else:
            grid_img[ini_r:r_step,ini_c:c_step]=black_block
            grid_map[i][j]=0

        ini_c=c_step
        c_step+=85

    ini_r=r_step
    r_step+=60
    ini_c=0
    c_step=85

loop_end_time=time.time()

#print(grid_map)
for i in grid_map:
    for j in i:
        print(int(j),end='  ')
    print()

code_end_time=time.time()

print("Loop execution time  %s seconds ---" % (loop_end_time - loop_start_time))
print("Code execution time  %s seconds ---" % (code_end_time - code_start_time))

cv2.imshow("Grid Image",grid_img)
cv2.imshow("Mask Image",mask)
cv2.waitKey(0)