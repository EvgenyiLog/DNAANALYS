#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
from wolframclient.language import wl, wlexpr
from wolframclient.evaluation import WolframLanguageSession 
def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-10-09_21-50-34.jpg")
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #session = WolframLanguageSession()
    session = WolframLanguageSession("D:/Program Files/Wolfram Research/Mathematica/13.3/WolframKernel.exe")
    image = session.evaluate(wl.RemoveBackground(image))
    print(image)
    #image=removebackgroud(image)
    session.terminate()
    #image=np.resize(image,(image.shape[1],image.shape[2]))
    #image=np.asarray(image,dtype=np.uint8)
    #plt.figure(figsize=(15, 7))
    #plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    #.tick_params(labelsize =20,#  Размер подписи
                    #color = 'k')   #  Цвет делений
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




