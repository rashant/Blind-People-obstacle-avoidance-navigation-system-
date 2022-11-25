import os
import cv2

image_path=os.path.join('D:\Projects\Trinetra Datasets\Dataset\Images',os.listdir('D:\Projects\Trinetra Datasets\Dataset\Images')[1])
masks_path=os.path.join('D:\Projects\Trinetra Datasets\Dataset\Masks',os.listdir('D:\Projects\Trinetra Datasets\Dataset\Masks')[1])
    
img=cv2.imread(image_path)
mask=cv2.imread(masks_path)

img=img/255
mask=mask/255

img = cv2.GaussianBlur(img, (7, 7), 0)
mask = cv2.GaussianBlur(mask, (7, 7), 0)

cv2.imshow("img",img)
cv2.imshow("mask",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()