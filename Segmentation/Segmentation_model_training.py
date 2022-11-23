import os
import pickle
import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import VGG16
from UNETMODEL import UNET
from keras.optimizers import Adam

x,y=[],[]

for i in os.listdir("D:\Projects\Trinetra Datasets\Dataset_shortened_2"):
    print(i)
    path=os.path.join("D:\Projects\Trinetra Datasets\Dataset_shortened_2",i)
    img=cv2.imread(path)
    img=img/255
    x.append(img)

    path=os.path.join("D:\Projects\Trinetra Datasets\Dataset\Masks",i)
    img=cv2.imread(path)
    img=img/255
    y.append(y)

# i,j=0,0
# while(i!=300):
#         print(i)
#         path=os.path.join("Dataset/Images",os.listdir("Dataset/Images")[i])
#         img=cv2.imread(path)
#         img=img/255
#         x.append(img)

#         path=os.path.join("Dataset/Masks",os.listdir("Dataset/Masks")[i])
#         img=cv2.imread(path)
#         img=img/255
#         y.append(img)
#         i+=1

with open("images_shortened2.pkl","wb") as file:
    pickle.dump(x,file)
print("images dumped")
with open("masks_shortened2.pkl","wb") as file:
    pickle.dump(y,file)
print("masks dumped")

# with open("images_shortened2.pkl","rb") as file:
#     x=pickle.load(file)
# print("images loaded")

# with open("masks_shortened2.pkl","rb") as file:
#     y=pickle.load(file)
# print("masks loaded")


# print("Loading model\n\n")
# model=UNET()
print("Loading pre-model\n\n")
model = VGG16(weights='imagenet', include_top=False,input_shape=(256,256,3))
print(model.summary())

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

optimizer=Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=[jaccard_distance])

colors = [ [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
class_names = ["road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

logs="Logs"
tensorboard_callback=TensorBoard(log_dir=logs)
x=np.array(x)
y=np.array(y)
print("Starting Training\n\n")
hist=model.fit(x,y,epochs=100)
print("Saving history\n\n")
with open("history.pkl","wb") as file:
    pickle.dump(hist,file)
model.save("dummy.h5")