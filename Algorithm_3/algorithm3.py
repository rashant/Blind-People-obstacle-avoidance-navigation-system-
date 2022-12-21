import numpy as np
import cv2
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose, InputLayer, Layer, Input, Dropout, MaxPool2D, concatenate
from keras.optimizers import Adam
import keras

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


class EncoderLayerBlock(Layer):
    def __init__(self, filters, rate, pooling=True):
        super(EncoderLayerBlock, self).__init__()
        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.c1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(self.rate)
        self.c2 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool = MaxPool2D(pool_size=(2,2))

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else: 
            return x

    def get_config(self):
        base_estimator = super().get_config()
        return {
            **base_estimator,
            "filters":self.filters,
            "rate":self.rate,
            "pooling":self.pooling
        }

#  Decoder Layer
class DecoderLayerBlock(Layer):
    def __init__(self, filters, rate, padding='same'):
            super(DecoderLayerBlock, self).__init__()
            self.filters = filters
            self.rate = rate
            self.cT = Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding=padding)
            self.next = EncoderLayerBlock(self.filters, self.rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.cT(X)
        c1 = concatenate([x, skip_X])
        y = self.next(c1)
        return y 

    def get_config(self):
        base_estimator = super().get_config()
        return {
            **base_estimator,
            "filters":self.filters,
            "rate":self.rate,
        }


def segmentation(frame):
        # Input Layer 
    input_layer = Input(shape=(256,256,3))

    # Encoder
    p1, c1 = EncoderLayerBlock(16,0.1)(input_layer)
    p2, c2 = EncoderLayerBlock(32,0.1)(p1)
    p3, c3 = EncoderLayerBlock(64,0.2)(p2)
    p4, c4 = EncoderLayerBlock(128,0.2)(p3)

    # Encoding Layer
    c5 = EncoderLayerBlock(256,0.3,pooling=False)(p4)

    # Decoder
    d1 = DecoderLayerBlock(128,0.2)([c5, c4])
    d2 = DecoderLayerBlock(64,0.2)([d1, c3])
    d3 = DecoderLayerBlock(32,0.2)([d2, c2])
    d4 = DecoderLayerBlock(16,0.2)([d3, c1])

    # Output layer
    output = Conv2D(3,kernel_size=1,strides=1,padding='same',activation='softmax')(d4)

    # U-Net Model
    model = keras.models.Model(
        inputs=[input_layer],
        outputs=[output],
        name="Segm"
    )

    # Compiling
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]
    )

    model.load_weights("D:\Projects\Trinetra\model_68.h5")
    im=np.expand_dims(frame,axis=0)
    pred=model.predict(im)
    return frame[0]

def path(frame):
    frame=cv2.resize(frame,(256,256))
    '''Cosidering only bottom part of the image as we don't need to analyze the complete image'''
    framex=segmentation(frame)
    
    framex=framex[180:,:]
    img=framex

    '''Creating masks'''
    # lower=np.array([146,  37,  93])
    # upper=np.array([157, 107, 150])

    # lower=np.array([0,  0,  47])
    # upper=np.array([179,  58, 183])

    lower=np.array([127,  0,  0])
    upper=np.array([168,  255, 127])

    GRAYTOBGR=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    imgHsv=cv2.cvtColor(GRAYTOBGR,cv2.COLOR_BGR2HSV)
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