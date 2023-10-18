#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from numpy.fft import fft, ifft
from matplotlib import pyplot as plt

def corrimage(image1,image2):
    dataFT1 = fft(image1, axis=1)
    dataFT2 = fft(image2, axis=1)
    dataC = ifft(dataFT1 * np.conjugate(dataFT2), axis=1).real
    plt.figure(figsize=(15, 7))
    plt.imshow(dataC, cmap=plt.cm.gray,vmax=dataC.max(),vmin=dataC.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    indexx,indexy=np.unravel_index(dataC.argmax(), dataC.shape)
    print(indexx)
    print(indexy)
    return indexx,indexy
    
import cv2    
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
    image1=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg") 
    image2=readimage("C:/Users/evgen/Downloads/s_1_1102_a.jpg") 
    indexx,indexy=corrimage(image1,image2)
    image1=image1[:indexx,:indexy]
    image2=image2[:indexx,:indexy]
    image21=image2[0:500,0:500]
    image11=image1[0:500,0:500]
    # concatenate image Horizontally 
    Hori = np.concatenate((image11, image21), axis=1) 
    plt.figure(figsize=(15,7))
    plt.imshow(Hori,cmap='gray',vmax=Hori.max(),vmin=Hori.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
  
    # concatenate image Vertically 
    Verti = np.concatenate((image11, image21), axis=0)
    plt.figure(figsize=(15,7))
    plt.imshow(Verti,cmap='gray',vmax=Verti.max(),vmin=Verti.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.show()
    
    
if __name__ == "__main__":
    main()
   


# In[ ]:




