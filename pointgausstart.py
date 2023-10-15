#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from scipy.ndimage import gaussian_filter,percentile_filter
import matplotlib.colors as colors
from matplotlib.patches import Circle
import imageio
from scipy import signal
import cv2

# Importing Numpy package
import numpy as np
 
# sigma(standard deviation) and muu(mean) are the parameters of gaussian
 
def gaussuian_filter(kernel_size, sigma=1, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    return gauss

def model():
    im2=np.random.randint(100, size=(256, 256))
    im3=np.random.random_sample((256, 256))
    #im=np.resize(im1+im2+im3,(256,256))
    #im = imageio.imread("F:\depositphotos_128127458-stock-illustration-circle-halftone-pattern.png")

    im=np.outer(signal.gaussian(512, 5), signal.gaussian(512, 5))#+np.random.random_sample((256, 256))
    for i in range(250):
        im=im+np.outer(signal.gaussian(512, i), signal.gaussian(512, i))
    #im=np.outer(signal.gaussian(256, 40), signal.gaussian(256, 40))+np.outer(signal.gaussian(256, 20), signal.gaussian(256, 20))+np.outer(signal.gaussian(256, 30), signal.gaussian(256, 30))+np.outer(signal.gaussian(256, 10), signal.gaussian(256, 10))+np.random.random_sample((256, 256))
    #im *= 1.0 / im.max()

    #im = gaussian_filter(im, 40)#+percentile_filter(im, percentile=20, size=20)-scipy.ndimage.sobel(im)-scipy.ndimage.uniform_filter(im, size=20)+scipy.ndimage.maximum_filter(im, size=20)+scipy.ndimage.minimum_filter(im, size=20)-scipy.ndimage.rank_filter(im, rank=42, size=20)-scipy.ndimage.prewitt(im)-scipy.ndimage.laplace(im)-scipy.ndimage.median_filter(im, size=20)
    print(im.shape)
    print(im.dtype)
    print(np.amin(im))
    print(np.amax(im))
    im=np.asarray(im,dtype=np.uint8)
    plt.figure(figsize=(15,7))
    plt.imshow(im,cmap='gray',vmax=im.max(),vmin=im.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    hist = cv2.calcHist([im], [0], None, [im.shape[1]], [0, im.shape[1]])
    hist=hist/hist.sum()
    plt.figure(figsize=(15,7))
    
    plt.plot(hist.flatten())
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    xx, yy = np.ogrid[0:im.shape[0],0:im.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d                                                                                                              fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx,yy,im)
    ax.grid(True)
    
    im= gaussuian_filter(5)
    plt.figure(figsize=(15,7))
    plt.imshow(im,cmap='gray',vmax=im.max(),vmin=im.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    xx, yy = np.ogrid[0:im.shape[0],0:im.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d                                                                                                              fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx,yy,im)
    ax.grid(True)
    hist=hist/hist.sum()
    plt.figure(figsize=(15,7))
    
    plt.plot(hist.flatten())
    
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    
def main():
    imagem=model()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()
                                                                              
                                                                                                                  


# In[ ]:




