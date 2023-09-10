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
    print(image.dtype)
    return image

def tiffreader(path):
    image=cv2.imread(path,-1)
    #print(image.shape)
    print(image.dtype)
    return image


def blendingimage(image1,image2,alpha=0.5,beta=0.5):
    max1=np.unravel_index(image1.argmax(), image1.shape)

    max1=np.asarray(max1,dtype=np.int64) #shift image of max?
    
    max2=np.unravel_index(image2.argmax(), image2.shape)

    max2=np.asarray(max2,dtype=np.int64)

    #image1=image1[::int(max1[0]),::int(max1[1])]
    #image2=image2[::int(max2[0]),::int(max2[1])]
    w, h = image2.shape[::-1]
    image1=cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
    image2=cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
    res = cv2.matchTemplate(image1,image2,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    imageblend = cv2.addWeighted(image1, alpha, image2, beta, 0.0)
    #imageblend=cv2.equalizeHist(imageblend)
    #clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(3,3))
    #imageblend = clahe.apply(imageblend)
    imageblend=cv2.normalize(imageblend, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U) 
    plt.figure(figsize=(15,7))
    plt.imshow(imageblend[150:200,150:200],cmap='gray',vmax=imageblend.max(),vmin=imageblend.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    
def main():
    image1=tiffreader("C:/Users/evgen/Downloads/s_1_1102_c.tif")
    image2=tiffreader("C:/Users/evgen/Downloads/s_1_1102_a.tif")
    image1=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    image2=readimage("C:/Users/evgen/Downloads/s_1_1102_a.jpg")
    blendingimage(image1,image2)
    plt.show()
    
    
if __name__ == "__main__":
    main()
    
    

    


# In[ ]:





# In[ ]:




