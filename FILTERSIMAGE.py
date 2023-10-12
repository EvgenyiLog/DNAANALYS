#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[12]:


import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def readimage(path):
    'path to file'
    'чтение файла возвращает изображение'
    
    image = cv2.imread(path,0)
    #print(image.shape)
    h, w = image.shape[:2]
    print(f'Weigth= {w}')
    print(f'Heigth= {h}')
    print(image.dtype)
    return image

def tiffreader(path):
    image=cv2.imread(path,-1)
    #print(image.shape)
    print(image.dtype)
    h, w = image.shape[:2]
    print(f'Weigth= {w}')
    print(f'Heigth= {h}')
    return image

def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-10-09_21-50-34.jpg")
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    # noise removal
    plt.figure(figsize=(15, 7))
    plt.imshow(thresh, cmap=plt.cm.gray,vmax=thresh.max(),vmin=thresh.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=5)
    plt.figure(figsize=(15, 7))
    plt.imshow(sure_bg, cmap=plt.cm.gray,vmax=sure_bg.max(),vmin=sure_bg.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    plt.figure(figsize=(15, 7))
    plt.imshow(sure_fg, cmap=plt.cm.gray,vmax=sure_fg.max(),vmin=sure_fg.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    unknown = cv2.subtract(sure_bg,sure_fg)
    plt.figure(figsize=(15, 7))
    plt.imshow(unknown, cmap=plt.cm.gray,vmax=unknown.max(),vmin=unknown.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    image=cv2.normalize(image, None, 0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]
    
    plt.show()
    
    
if __name__ == "__main__":
    main()
    


# In[ ]:




