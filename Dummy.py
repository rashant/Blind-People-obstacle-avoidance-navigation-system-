# model=UNET((256,256,3))
#
# def jaccard_distance_loss(y_true,y_pred,smooth=100):
#     intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
#     Union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
#     jac=(intersection+smooth)/(Union-intersection+smooth)
#     return (1-jac)*smooth
#
# def dice_metric(y_true,y_pred):
#     intersection=K.sum(K.sum(K.abs(y_true*y_pred),axis=-1))
#     Union=K.sum(K.sum(K.abs(y_true)+K.abs(y_pred),axis=-1))
#
#     return 2*intersection/Union
#
# model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=[dice_metric])
#
# print(model.summary())
#
# steps_per_epoch= len(os.listdir("train")) // 32
#
# hist=model.fit(train_generator,validation_data=val_generator,epochs=50,
#                steps_per_epoch=32,
#                validation_steps=32)
#
# model.save("dummy.h5")
#
# with open("history.pkl","wb") as file:
#     pickle.dump(hist,file)
from keras import backend as K
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

from keras.models import load_model
model=load_model("dummy.h5",custom_objects={"jaccard_distance":[jaccard_distance]})
import cv2
import numpy as np
img=cv2.imread("Dataset/Images/image_16.jpg")
img=img/255
print(np.array(img).shape)
prediction=model.predict(np.expand_dims(np.array(img),axis=0))
print(prediction)
import matplotlib.pyplot as plt
print(prediction[0].shape)
plt.imshow(prediction[0])
plt.show()