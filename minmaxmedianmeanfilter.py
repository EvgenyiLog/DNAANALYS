#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
import mahotas


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
    
    
def maximage(image):
    imagemax=scipy.ndimage.maximum_filter(image,size=3)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagemax[0:1000,0:1000], cmap=plt.cm.gray,vmax=imagemax.max(),vmin=imagemax.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    xx, yy = np.ogrid[0:imagemax.shape[0],0:imagemax.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,imagemax)
    ax.set_zlim(0,50)
    ax.grid(True)
    
    
    
def minimage(image):
    imagemin=scipy.ndimage.minimum_filter(image,size=3)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagemin[0:1000,0:1000], cmap=plt.cm.gray,vmax=imagemin.max(),vmin=imagemin.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    xx, yy = np.ogrid[0:imagemin.shape[0],0:imagemin.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,imagemin)
    ax.set_zlim(0,50)
    ax.grid(True)
    
def medianimage(image):
    imagemedian=scipy.ndimage.median_filter(image,size=3)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagemedian[0:1000,0:1000], cmap=plt.cm.gray,vmax=imagemedian.max(),vmin=imagemedian.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    xx, yy = np.ogrid[0:imagemedian.shape[0],0:imagemedian.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,imagemedian)
    ax.set_zlim(0,50)
    ax.grid(True)
    
def meanfilter(image):
    imagemean=mahotas.mean_filter(image, 3)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagemean[0:1000,0:1000], cmap=plt.cm.gray,vmax=imagemean.max(),vmin=imagemean.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    xx, yy = np.ogrid[0:imagemean.shape[0],0:imagemean.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,imagemean)
    ax.set_zlim(0,50)
    ax.grid(True)
    
def main():
    image=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    maximage(image)
    minimage(image)
    medianimage(image)
    meanfilter(image)
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




