#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.ndimage import generic_filter
from pylab import *
import scipy
def   percentileFilter(f,p,K):
    def pp(v):
        return percentile(v, 100*p)
    return generic_filter(f, pp, size=(K,K))


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
    
    # Apply kernel for embossing 
    emboss_kernel = np.array([[-1, 0, 0], 
                    [0, 0, 0], 
                    [0, 0, 1]]) 
  
    # Embossed image is obtained using the variable emboss_img 
    # cv2.fliter2D() is the function used 
    # src is the source of image(here, img) 
    # ddepth is destination depth. -1 will mean output image will have same depth as input image 
    # kernel is used for specifying the kernel operation (here, emboss_kernel) 
    emboss_img = cv2.filter2D(src=image, ddepth=-1, kernel=emboss_kernel)
    plt.figure(figsize=(15, 7))
    plt.imshow(emboss_img,cmap=plt.cm.gray,vmax=emboss_img.max(),vmin=emboss_img.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    # Apply kernel for sharpening 
    sharp_kernel = np.array([[0, -1, 0], 
                    [-1, 5, -1], 
                    [0, -1, 0]]) 
  
    # Sharpeneded image is obtained using the variable sharp_img 
    # cv2.fliter2D() is the function used 
    # src is the source of image(here, img) 
    # ddepth is destination depth. -1 will mean output image will have same depth as input image 
    # kernel is used for specifying the kernel operation (here, sharp_kernel) 
    sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=sharp_kernel) 
  
    # Showing the sharpened image using matplotlib library function plt.imshow() 
    plt.figure(figsize=(15, 7))
    plt.imshow(sharp_img,cmap=plt.cm.gray,vmax=sharp_img.max(),vmin=sharp_img.min()) 
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    imagegl=scipy.ndimage.gaussian_laplace(image, sigma=3)
    print('min')
    print(np.amin(imagegl))
    print('max')
    print(np.amax(imagegl))
    plt.figure(figsize=(15, 7))
    plt.imshow(imagegl,cmap=plt.cm.gray,vmax=imagegl.max(),vmin=imagegl.min()) 
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    imagep=percentileFilter(image,0.5,9)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagep,cmap=plt.cm.gray,vmax=imagep.max(),vmin=imagep.min()) 
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.show()
    
    
if __name__ == "__main__":
    main()
    


# In[ ]:




