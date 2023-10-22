#!/usr/bin/env python
# coding: utf-8

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
''''
def update_map(ind, map_x, map_y):
    if ind == 0:
    for i in range(map_x.shape[0]):
    for j in range(map_x.shape[1]):
    if j > map_x.shape[1]*0.25 and j < map_x.shape[1]*0.75 and i > map_x.shape[0]*0.25 and i < map_x.shape[0]*0.75:
    map_x[i,j] = 2 * (j-map_x.shape[1]*0.25) + 0.5
    map_y[i,j] = 2 * (i-map_y.shape[0]*0.25) + 0.5
    else:
    map_x[i,j] = 0
    map_y[i,j] = 0
    elif ind == 1:
    for i in range(map_x.shape[0]):
    map_x[i,:] = [x for x in range(map_x.shape[1])]
    for j in range(map_y.shape[1]):
    map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]
    elif ind == 2:
    for i in range(map_x.shape[0]):
    map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
    for j in range(map_y.shape[1]):
    map_y[:,j] = [y for y in range(map_y.shape[0])]
    elif ind == 3:
    for i in range(map_x.shape[0]):
    map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
    for j in range(map_y.shape[1]):
    map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]
'''

def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-10-09_21-50-34.jpg")
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    map_x = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_y = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    rows, cols = image.shape
 
    pts1 = np.float32([[50, 50],
                   [200, 50], 
                   [50, 200]])
 
    pts2 = np.float32([[10, 100],
                   [200, 50], 
                   [100, 250]])
 
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    (cX, cY) = (rows // 2, cols // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
    image = cv2.warpAffine(image, M, (rows, cols))
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
     
    plt.show()
    
    
if __name__ == "__main__":
    main()
    
 


# In[ ]:




