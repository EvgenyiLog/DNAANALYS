#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image
from PIL import ImageOps
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
import keras
from skimage import color, data, restoration
from scipy.signal import convolve2d
import sporco
import skimage 
from scipy import ndimage
from skimage import measure
import matlab
import matlab.engine 
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
from wolframclient.language import wl, wlexpr
from wolframclient.evaluation import WolframLanguageSession 
#import htmlPy
import eel
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage import filters
from skimage.filters import try_all_threshold
from skimage import morphology
import pandas as pd
from pyclesperanto_prototype import imshow
import pyclesperanto_prototype as cle
from skimage.io import imread
from skimage.feature import peak_local_max
from skimage.filters import hessian
from skimage.metrics import peak_signal_noise_ratio
import pandas as pd
from skimage.filters.rank import entropy
from skimage.morphology import disk


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
def imageplot3d(image):
    #x = np.linspace(0, image.shape[1], image.shape[1])
    #y = np.linspace(0, image.shape[0], image.shape[0])
    # full coordinate arrays
    #xx, yy = np.meshgrid(x, y)
    xx, yy = np.ogrid[0:image.shape[0],0:image.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,image)
    ax.grid(True)
    
    
from prettytable import PrettyTable
from colorama import init, Fore, Back, Style     
def localstdmean(image,N):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    noise=np.sqrt((s2 - s**2 / ns) / ns)
    noise=np.asarray(noise,dtype=np.uint8)
    plt.figure(figsize=(15, 7))
    plt.imshow(noise[0:1000,0:1000], cmap=plt.cm.gray,vmax=noise.max(),vmin=noise.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    noises=cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_noise.jpg",noises)
    background=scipy.signal.convolve2d(im, kernel, mode="same")
    background=np.asarray(background,dtype=np.uint8)
    backgrounds=cv2.cvtColor( background, cv2.COLOR_GRAY2BGR) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_background.jpg",backgrounds)
    plt.figure(figsize=(15, 7))
    plt.imshow(background[0:1000,0:1000], cmap=plt.cm.gray,vmax=background.max(),vmin=background.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    print(f'image min ={image.min()}')
    print(f'image max ={image.max()}')
    print(f'image mean ={image.mean()}')
    print(f'image std ={image.std()}')
    
    
def cohernceimage(image1,image2):
    # Compute the discrete Fourier Transform of the image
    fourier1 = cv2.dft(np.float32(image1), flags=cv2.DFT_COMPLEX_OUTPUT)
 
    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift1 = np.fft.fftshift(fourier1)
 
    # calculate the magnitude of the Fourier Transform
    magnitude1 = cv2.magnitude(fourier_shift1[:,:,0],fourier_shift1[:,:,1])
    # Compute the discrete Fourier Transform of the image
    fourier2 = cv2.dft(np.float32(image2), flags=cv2.DFT_COMPLEX_OUTPUT)
 
    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift2 = np.fft.fftshift(fourier2)
 
    # calculate the magnitude of the Fourier Transform
    magnitude2 = cv2.magnitude(fourier_shift2[:,:,0],fourier_shift2[:,:,1])
    coherence=np.divide(np.square(np.abs(np.multiply(fourier1,fourier2.conj()))),
                        np.multiply(np.square(np.abs(fourier1)),np.square(np.abs(fourier2))))
    print(coherence.shape)
    coherencex=coherence[:,:,0]
    plt.figure(figsize=(15, 7))
    plt.imshow(coherencex[0:1000,0:1000], cmap=plt.cm.gray,vmax=coherencex.max(),vmin=coherencex.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    xx, yy = np.ogrid[0:coherencex.shape[0],0:coherencex.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,coherencex)
    
    ax.grid(True)
    plt.savefig("C:/Users/evgen/Downloads/3Dcoherencex.jpg")
    
    
    coherencey=coherence[:,:,0]
    plt.figure(figsize=(15, 7))
    plt.imshow(coherencey[0:1000,0:1000], cmap=plt.cm.gray,vmax=coherencey.max(),vmin=coherencey.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    xx, yy = np.ogrid[0:coherencey.shape[0],0:coherencey.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,coherencey)
    
    ax.grid(True)
    plt.savefig("C:/Users/evgen/Downloads/3Dcoherencey.jpg")
    
def main():
    image1=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    image2=readimage("C:/Users/evgen/Downloads/s_1_1102_a.jpg")
    cohernceimage(image1,image2)
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:



