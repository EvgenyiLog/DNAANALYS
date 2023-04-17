#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
    image=np.random.randint(0, 2, size = (2944, 2944),dtype=int)
    print(image.shape)
    print(image)
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    image=np.linalg.inv(image)
    image=np.asarray(image,dtype=int)
    print(image)
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    
    
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()


# In[ ]:




