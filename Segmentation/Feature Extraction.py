from keras.applications.vgg16 import VGG16
import cv2
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=False,input_shape=(256,256,3))
#model = VGG16(weights='imagenet', include_top=False)
print(model.summary())
new_model=Model(model.input,model.get_layer('block1_conv2').output)

img_path = 'D:\Projects\Trinetra Datasets\Dataset_shortened_2\image_355.jpg'
img=cv2.imread(img_path)
x=np.array(img)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = new_model.predict(x)
print(features.shape)
square=4
ix=1

for _ in range(square):
    for _ in range(square):
        ax=plt.subplot(square,square,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1])
        ix+=1
plt.tight_layout()
plt.show()
idx=ix
square=4
ix=1
for _ in range(square):
    for _ in range(square):
        ax=plt.subplot(square,square,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,idx-1])
        ix+=1
        idx+=1
plt.tight_layout()
plt.show()

square=4
ix=1
for _ in range(square):
    for _ in range(square):
        ax=plt.subplot(square,square,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,idx-1])
        ix+=1
plt.tight_layout()
plt.show()

square=4
ix=1
for _ in range(square):
    for _ in range(square):
        ax=plt.subplot(square,square,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,idx-1])
        ix+=1
plt.tight_layout()
plt.show()
