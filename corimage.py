#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
import scipy
from skimage import color, data, restoration
from scipy.signal import convolve2d
import sporco
import skimage 
from scipy import ndimage
from skimage import measure
from scipy import signal



import numpy as np
from numpy.fft import fft2, ifft2

def wiener_filter(img, kernel, K = 10):
    dummy = np.copy(img)
    kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
    # Fourier Transform
    dummy = fft2(dummy)
    kernel = fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return np.uint8(dummy)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)



def gamma_trans(img,gamma=1.0):
    # Конкретный метод сначала нормализуется до 1, а затем гамма используется в качестве значения индекса, чтобы найти новое значение пикселя, а затем восстановить
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # Реализация сопоставления использует функцию поиска в таблице Opencv
    return cv2.LUT(img,gamma_table)

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
def main():
    image = cv2.imread("C:/Users/evgen/Downloads/s_1_1102_c.jpg",1)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ''''
    corr1 = signal.correlate2d(image[:,:,0], image[:,:,1], boundary='symm', mode='same')
    plt.figure(figsize=(15,7))
    plt.imshow(cor1)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    corr2 = signal.correlate2d(image[:,:,0], image[:,:,2], boundary='symm', mode='same')
    plt.figure(figsize=(15,7))
    plt.imshow(cor2)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    '''
    
    mse_none = mean_squared_error(image[:,:,0], image[:,:,1])
    ssim_none = ssim(image[:,:,0],image[:,:,1], data_range=image[:,:,0].max() - image[:,:,1].min())
    print(ssim_none)
    
    image=scipy.signal.wiener(image)
    print(image.shape)
    plt.figure(figsize=(15,7))
    plt.imshow(image)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    kernel = np.array([[0, -1, 0], [-1, 6.5, -1], [0, -1, 0]])
  

    image = cv2.filter2D(image, -1, kernel)
    plt.figure(figsize=(15,7))
    plt.imshow(image)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    #image=adjust_gamma(image)
    #image=gamma_trans(image)
    plt.figure(figsize=(15,7))
    plt.imshow(image)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    image = cv2.GaussianBlur(image,(3,3),0)
    
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    #image= cv2.medianBlur(image,3)
    #plt.figure(figsize=(15,7))
    #plt.imshow(image,cmap='gray')
    #plt.grid(True)
    #plt.tick_params(labelsize =20,#  Размер подписи
                    #color = 'k')   #  Цвет делений
    
    #image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.imread("C:/Users/evgen/Downloads/s_1_1102_c.jpg",0)
    
    slp,hlp=sporco.signal.tikhonov_filter(image,2)
    slp=cv2.cvtColor(slp, cv2.COLOR_GRAY2BGR)
    hlp=cv2.cvtColor(hlp, cv2.COLOR_GRAY2BGR)
    slp=cv2.cvtColor(slp, cv2.COLOR_BGR2RGB)
    hlp=cv2.cvtColor(hlp, cv2.COLOR_BGR2RGB)
   
    plt.figure(figsize=(15,7))
    
    plt.imshow(slp)
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    plt.figure(figsize=(15,7))
    
    plt.imshow(hlp)
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




