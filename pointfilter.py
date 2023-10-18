#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy
from numpy.fft import fft, ifft
from matplotlib import pyplot as plt
import numpy as np


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

from scipy import signal
def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-10-09_21-50-34.jpg")
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    
    
    im=np.outer(signal.gaussian(2, 5), signal.gaussian(2, 5))#+np.random.random_sample((256, 256))
    for i in range(2):
        im=im+np.outer(signal.gaussian(2, i), signal.gaussian(2, i))
        
    im=np.asarray(im,dtype=np.uint8)
    
    imageg = cv2.filter2D(image,-1,im)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(imageg, cmap=plt.cm.gray,vmax=imageg.max(),vmin=imageg.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    im=np.outer(signal.windows.hann(10),signal.windows.hann(10))
    im=np.asarray(im,dtype=np.uint8)
    imageg = cv2.filter2D(image,-1,im)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(imageg, cmap=plt.cm.gray,vmax=imageg.max(),vmin=imageg.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    im=np.outer(signal.windows.gaussian(5, std=7),signal.windows.gaussian(5, std=7))
    imageg = cv2.filter2D(image,-1,im)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(imageg, cmap=plt.cm.gray,vmax=imageg.max(),vmin=imageg.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    im=np.outer(signal.windows.general_gaussian(5, p=1.5, sig=7),signal.windows.general_gaussian(5, p=1.5, sig=7))
    im=np.asarray(im,dtype=np.uint8)
    imageg = cv2.filter2D(image,-1,im)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(imageg, cmap=plt.cm.gray,vmax=imageg.max(),vmin=imageg.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




