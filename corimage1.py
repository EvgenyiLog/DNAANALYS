#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
def main():
    image1 = cv2.imread("C:/Users/evgen/Downloads/s_1_1102_c.jpg",0)
    plt.figure(figsize=(15,7))
    plt.imshow(image1,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    image2 = cv2.imread("C:/Users/evgen/Downloads/s_1_1101_a.jpg",0)
    plt.figure(figsize=(15,7))
    plt.imshow(image2,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    mse = mean_squared_error(image1, image2)
    ssim = structural_similarity(image1, image2, data_range=image2.max() - image2.min())
    print(mse)
    print(ssim)
    
    '''
    corr1 = signal.correlate2d(image1, image2, boundary='symm', mode='same')
    plt.figure(figsize=(15,7))
    plt.imshow(cor1)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    corr2 = signal.correlate2d(image2, image1, boundary='symm', mode='same')
    plt.figure(figsize=(15,7))
    plt.imshow(cor2)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    '''
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




