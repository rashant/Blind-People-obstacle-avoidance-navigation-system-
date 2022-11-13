import os
import cv2

# Dataset='Val_Images/val'
# count=2501


# print(f"Total length of dataset is {len(os.listdir(Dataset))}")
# for i in os.listdir(Dataset):
#     image_path=os.path.join(Dataset,i)
#     try:
#         image=cv2.imread(image_path)
#         cv2.imwrite(f"Dataset/Images/image_{count}.jpg",image)
#         count+=1
#         print(f"Image {count}")
#     except Exception as e:
#         pass
count=0
for i in os.listdir('train'):
    path=os.join('train/'+i)
    image=cv2.imread(path)
    img=image[:,:256,:]
    mask=image[:,256:,:]

    cv2.imwrite(f"Dataset/Images/image_{count}.jpg",img)
    cv2.imwrite(f"Dataset/Masks/image_{count}.jpg",mask)
    count+=1

count=0
for i in os.listdir('val'):
    path=os.join('val/'+i)
    image=cv2.imread(path)
    img=image[:,:256,:]
    mask=image[:,256:,:]

    cv2.imwrite(f"Dataset/Images/image_{count}.jpg",img)
    cv2.imwrite(f"Dataset/Masks/image_{count}.jpg",mask)
    count+=1