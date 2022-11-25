# Common
import os
import keras
import numpy as np
import cv2
from UNETMODEL import *

# Data Viz
import matplotlib.pyplot as plt

# Model 
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose, InputLayer, Layer, Input, Dropout, MaxPool2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

x,y=[],[]

for i in range(300):
    image_path=os.path.join('D:\Projects\Trinetra Datasets\Dataset\Images',os.listdir('D:\Projects\Trinetra Datasets\Dataset\Images')[i])
    masks_path=os.path.join('D:\Projects\Trinetra Datasets\Dataset\Masks',os.listdir('D:\Projects\Trinetra Datasets\Dataset\Masks')[i])
    
    img=cv2.imread(image_path)
    mask=cv2.imread(masks_path)

    img=img/255
    mask=mask/255

    img = cv2.GaussianBlur(img, (1, 1), 0)
    mask = cv2.GaussianBlur(mask, (1, 1), 0)

    x.append(img)
    y.append(mask)

x=np.array(x)
y=np.array(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

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

#  Callback 
class ShowProgress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(x_test))
        rand_img = x_test[id][np.newaxis,...]
        pred_mask = self.model.predict(rand_img)[0]
        true_mask = y_test[id]


        plt.subplot(1,3,1)
        plt.imshow(rand_img[0])
        plt.title("Original Image")
        plt.axis('off')


        plt.subplot(1,3,2)
        plt.imshow(pred_mask)
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(true_mask)
        plt.title("True Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Input Layer 
input_layer = Input(shape=x_train.shape[-3:])

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
output = Conv2D(3,kernel_size=1,strides=1,padding='same',activation='sigmoid')(d4)

    # U-Net Model
model = build_vgg16_unet((256,256,3))

    # Compiling
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy', keras.metrics.MeanIoU(num_classes=3)]
)

    # Callbacks 
callbacks =[
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("UNet-segmentizer.h5", save_best_only=True),
    ShowProgress()
]


    # Train The Model
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=500,
    callbacks=callbacks,
    steps_per_epoch=32
)