import numpy as np
import cv2

def pathMap(path_map,directions,white_block,black_block):
    r_start=160
    r_end=180
    c_start=180
    c_end=200

    for i in directions:
        match i:
            case 'S':
                path_map[r_start:r_end,c_start:c_end]=white_block
                r_end=r_start
                r_start-=20
            
            case 'L':
                c_end=c_start
                c_start-=20
                path_map[r_start:r_end,c_start:c_end]=white_block
                r_end=r_start
                r_start-=20

            case 'R':
                c_start=c_end
                c_end+=20
                path_map[r_start:r_end,c_start:c_end]=white_block
                r_end=r_start
                r_start-=20

            case 'ST':
                path_map=black_block


def place(grid,i,j):
    '''Checking if there is way ahead. We will move straight if we can take 2 steps forward'''
    try:
        if grid[i][j]==1 and grid[i-1][j]==1:
            return True
        return False
    except:
        return True

def backtrack(grid,directions):
    i=8
    j=9
    while True:
        status=0
        '''If the place function returns true that means we can move 2 steps ahead at current time frame. So we can move straight'''
        if(place(grid,i,j))==True:
            directions.append(0)
            status=1
            i=i-1
        
        '''If the place function returns false that means there is an obstacle a head at current time frame. So we need to move left or right'''
        if(status==0):
            l=grid[0:9,0:9]
            r=grid[0:9,9:18]
            '''Because the current step is a wrong step we will move backward and decide to go Left/Right'''
            i=i+1
            '''We will move left or right on the basis of maximum walkable area in the both sides. We compare if there is more walkable area 
            in the right we will move right else left. '''
            lc,rc=np.count_nonzero(l),np.count_nonzero(r)
            try:
                if rc>lc:
                    j=j+1
                    directions.pop()
                    directions.append(1)
                else:
                    j=j-1
                    directions.pop()
                    directions.append(2)
                '''If there is no way to move the direction will be empty and when we apply backtracking on it, it will raise error because there
                is we are already at the starting point. So we need to wait until the obstacle goes out of the way.'''
            except:
                directions.append(3)
                break
            print()
            '''We will stop the path finding process once we could find the next 5 steps'''
        if len(directions)==9:
            break

def path(frame):
    '''Cosidering only bottom part of the image as we don't need to analyze the complete image'''
    frame=frame[180:,:,:]
    img=frame

    '''Creating masks'''
    # lower=np.array([146,  37,  93])
    # upper=np.array([157, 107, 150])

    lower=np.array([0,  0,  47])
    upper=np.array([179,  58, 183])

    imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHsv,lower,upper)

    black_block= np.zeros([20, 20], dtype = np.float32)
    white_block= np.ones([20, 20], dtype = np.float32)

    '''Grid Image'''
    grid_img= np.zeros([180,360],dtype=np.float32)

    '''Matrix representation of grid image'''
    grid_map=np.zeros([9,18],dtype=np.float32)

    '''Path Image'''
    path_map=np.zeros([180,360],dtype=np.float32)

    '''Size of each block in the grid is 20X20 and size of grid is 9X18'''
    r_step=20
    c_step=20
    ini_r=0
    ini_c=0

    for i in range(9):
        for j in range(18):
            avg_list=[]
            sumx=0
            '''Picking each block from the mask'''
            image_block=mask[ini_r:r_step,ini_c:c_step]

            '''Finding the average of white space in each block'''
            for k in image_block:
                sumx+=(sum(k)/len(k))

            average=sumx/18
            avg_list.append(round(average,0))

            '''Assigning white block if the white area is greater than threshold'''
            if round(average,0)>=210:
                grid_img[ini_r:r_step,ini_c:c_step]=white_block
                grid_map[i][j]=1
            else:
                grid_img[ini_r:r_step,ini_c:c_step]=black_block
                grid_map[i][j]=0

            ini_c=c_step
            c_step+=20

        ini_r=r_step
        r_step+=20
        ini_c=0
        c_step=20
    
    grid_map=np.array(grid_map)

    directions=[]

    '''Applying backtracking for path finding'''
    backtrack(grid_map,directions)

    '''Applying path over the path map'''
    pathMap(path_map,directions,white_block,black_block)


    return grid_img,mask,grid_map,path_map,directions
    #return grid_img,mask,directions