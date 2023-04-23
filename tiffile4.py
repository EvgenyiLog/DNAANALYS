#!/usr/bin/env python
# coding: utf-8

# In[36]:


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

def main():
    with Image.open("C:/Users/evgen/Downloads/s_1_1102_c.jpg") as img:
        img.load()
        
        
    #img.show()
    plt.figure(figsize=(15,7))
    plt.imshow(img,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    image = cv2.imread("C:/Users/evgen/Downloads/s_1_1102_c.jpg",0)
    print(image.shape)
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    kernel = np.array([[0, -1, 0], [-1, 6.5, -1], [0, -1, 0]])
  

    image = cv2.filter2D(image, -1, kernel)
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    img_blur = cv2.GaussianBlur(image,(1,1),0) 
    plt.figure(figsize=(15,7))
    plt.imshow(img_blur,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
 
    plt.figure(figsize=(15,7))
    plt.subplot(2,2,1)
    plt.imshow(img_blur,cmap = 'gray')
    plt.title('Оригинал',fontsize=20)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.subplot(2,2,2)
    plt.imshow(sobelxy,cmap = 'gray')
    plt.title('Собель XY',fontsize=20)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.subplot(2,2,3)
    plt.imshow(sobelx,cmap = 'gray')
    plt.title('Собель X',fontsize=20)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.subplot(2,2,4)
    plt.imshow(sobely,cmap = 'gray')
    plt.title('Собель Y',fontsize=20)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
    plt.figure(figsize=(15,7))
    
    plt.imshow(edges,cmap = 'gray')
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    psf = np.ones((5, 5)) / 25
    img = convolve2d(image, psf, 'same')
    rng = np.random.default_rng()
    img += 0.1 * img.std() * rng.standard_normal(img.shape)
    deconvolved_img = restoration.unsupervised_wiener(img, psf)
    plt.figure(figsize=(15,7))
    
    plt.imshow(img,cmap = 'gray')
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    slp,hlp=sporco.signal.tikhonov_filter(image,2)
    plt.figure(figsize=(15,7))
    
    plt.imshow(slp,cmap = 'gray')
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    plt.figure(figsize=(15,7))
    
    plt.imshow(hlp,cmap = 'gray')
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    edges = cv2.Canny(image=image-slp, threshold1=100, threshold2=200) 
    plt.figure(figsize=(15,7))
    
    plt.imshow(edges,cmap = 'gray')
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    # Remove small white regions
    open_img = ndimage.binary_opening(image)
    # Remove small black hole
    close_img = ndimage.binary_closing(image)
    
    
    plt.figure(figsize=(15,7))
    
    plt.imshow(close_img,cmap = 'gray')
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    eroded_img = ndimage.binary_erosion(image)
    reconstruct_img = ndimage.binary_propagation(eroded_img, mask=image)
    tmp = np.logical_not(reconstruct_img)
    eroded_tmp = ndimage.binary_erosion(tmp)
    reconstruct_final = np.logical_not(ndimage.binary_propagation(eroded_tmp, mask=tmp))
    
    
    
    # Find contours at a constant value of 0.05
    contours = measure.find_contours(image, 0.005)

    # Display the image and plot all contours found
    fig, ax = plt.subplots(figsize=(15,7))
    ax.imshow(image, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
                    
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()


# In[ ]:




