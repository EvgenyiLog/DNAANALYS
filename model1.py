#!/usr/bin/env python
# coding: utf-8

# In[103]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#from libtiff import TIFF
from skimage import io
#import pytiff
from tifffile import tifffile
#import OpenImageIO as oiio
#import rasterio
#import tensorflow_io as tfio
import cv2



def main():
    image=np.zeros((3000, 3000,3))
    
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    sample_x = np.random.normal(1500, 200, 95000000)
    sample_y = np.random.normal(1500, 200, 95000000)
    
    #print(sample_x)
    #print(sample_y)
    sample_x=np.trunc((image.shape[0])/(np.amax(sample_x)+200)*sample_x)
    sample_x=np.asarray(sample_x,dtype=int)
    sample_y=np.trunc((image.shape[1])/(np.amax(sample_y)+200)*sample_y)
    sample_y=np.asarray(sample_y,dtype=int)
    
    image[sample_x,sample_y,0]=200
    image[sample_x,sample_y,1]=200
    image[sample_x,sample_y,2]=200
    
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    image=np.zeros((3000, 3000,3))
    meanvalue=1500
    modevalue = np.sqrt(2 / np.pi) * meanvalue
    
    

    sample_x = np.random.rayleigh(modevalue,3000)
    print(sample_x)
    sample_y = np.random.rayleigh(modevalue,3000)
    print(sample_y)
    
    
    sample_x=np.trunc((image.shape[0])/(np.amax(sample_x)+1)*sample_x)
    sample_x=np.asarray(sample_x,dtype=int)
    sample_y=np.trunc((image.shape[1])/(np.amax(sample_y)+1)*sample_y)
    sample_y=np.asarray(sample_y,dtype=int)
    
    image[sample_x,sample_y,0]=255
    image[sample_x,sample_y,1]=255
    image[sample_x,sample_y,2]=255
          
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




