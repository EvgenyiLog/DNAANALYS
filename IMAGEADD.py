#!/usr/bin/env python
# coding: utf-8

# In[8]:


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
    image1=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    image2=readimage("C:/Users/evgen/Downloads/s_1_1102_a.jpg")
    image3=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    image4=readimage("C:/Users/evgen/Downloads/s_1_1102_a.jpg")
    image5=np.zeros_like(image1,dtype=np.uint8)
    image5=cv2.cvtColor(image5, cv2.COLOR_GRAY2RGB)
    #image5=cv2.cvtColor(image5, cv2.COLOR_RGB2CMYK)
    plt.show()
    
    
if __name__ == "__main__":
    main()
    


# In[ ]:




