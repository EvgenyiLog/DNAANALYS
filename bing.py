#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
#import sunkit_image

import numpy as np

def boxplot_2d(x,y, ax, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k'
    )



def generate_feature_stack(image):
    # determine features
    blurred = filters.gaussian(image, sigma=2)
    edges = filters.sobel(blurred)

    # collect features in a stack
    # The ravel() function turns a nD image into a 1-D image.
    # We need to use it because scikit-learn expects values in a 1-D format here. 
    feature_stack = [
        image.ravel(),
        blurred.ravel(),
        edges.ravel()
    ]
    
    # return stack as numpy-array
    return np.asarray(feature_stack)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def gamma_trans(img,gamma):
    # Конкретный метод сначала нормализуется до 1, а затем гамма используется в качестве значения индекса, чтобы найти новое значение пикселя, а затем восстановить
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # Реализация сопоставления использует функцию поиска в таблице Opencv
    return cv2.LUT(img0,gamma_table)


def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    return np.sqrt((s2 - s**2 / ns) / ns)

def laplacian_of_gaussian(image, sigma):
    """
    Applies a Gaussian kernel to an image and the Laplacian afterwards.
    """
    
    # blur the image using a Gaussian kernel
    intermediate_result = filters.gaussian(image, sigma)
    
    # apply the mexican hat filter (Laplacian)
    result = filters.laplace(intermediate_result)
    
    return result



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
    
    
def filtration(image):
    hist = cv2.calcHist([image],[0],None,[4096],[0,4096])
    print(np.amax(hist))
    print(np.argmax(hist))
    #print(type(hist))
    h=list(hist)
    index05max=[]
    for i in h:
        if i>=0.5*np.amax(hist):
            index05max.append(h.index(i))
    index99prec=[]
    for i in h:
        if i>=np.percentile(hist,99):
            index99prec.append(h.index(i))
    index05max=np.asarray(index05max,dtype=np.int64)
    index05max=np.asarray(index05max,dtype=np.int64)
    index99prec=np.asarray(index99prec,dtype=np.int64)
    #print(index05max.ravel())
    print(np.amax(index05max))
    print(np.amin(index05max))
    print(np.amax(index99prec))
    plt.figure(figsize=(15,7))
    plt.plot(hist)
    plt.axvline(x=np.argmax(hist), color='b', label='max')
    plt.axvline(x=np.amax(index05max), color='g', label='0.5*max')
    plt.axvline(x=np.amin(index05max), color='r', label='0.5*max')
    plt.axvline(x=np.amax(index99prec), color='m', label='99per')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                     color = 'k')   #  Цвет делений
    
    
    plt.figure()
    plt.imshow(image[0:500,0:500],cmap='gray',vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    boxplot_2d(image[0:int(image.shape[0]),:],image[:,0:int(image.shape[1])],ax=ax2, whis=7)
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #image=np.where(image<np.amax(index05max),image,0)
    image=np.where(image>np.amin(index05max),image,0)
    psnr=peak_signal_noise_ratio(image, std_convoluted(image,2))
    print(f'PSNR after filtration={psnr}')
    print(f'PSNR after filtration={20*np.log10(1/image.std())}')
    try:
        mse = mean_squared_error(image,image.T)
        print(f'PSNR after filtration={20*np.log10(1/mse)}')
    except:
        pass
    try:
        psnr=cv2.PSNR(image,image.T)
        print(f'PSNR after filtration={psnr}')
    except:
        pass
    
    print(f'PSNR after filtration max std={20*np.log10(image.max()/image.std())}')
    plt.figure()
    plt.imshow(image[0:100,0:100],cmap='gray',vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    print(np.amax(image))
    print(image.dtype)
    print(image.shape)
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    boxplot_2d(image[0:int(image.shape[0]),:],image[:,0:int(image.shape[1])],ax=ax2, whis=7)
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.figure()
    plt.imshow(image[0:500,0:500],cmap='gray',vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    return image

def clusters(image):
    image=cv2.GaussianBlur(image,(3,3),0)
    laplac=cv2.Laplacian(image,cv2.CV_16S, ksize=3)
    xx, yy = np.ogrid[0:laplac.shape[0],0:laplac.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,laplac)
    ax.grid(True)
    
    
    abs_laplac = cv2.convertScaleAbs(laplac)
    xx, yy = np.ogrid[0:abs_laplac.shape[0],0:abs_laplac.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,abs_laplac)
    ax.grid(True)
    
    
def main():
    image=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    #image=tiffreader("C:/Users/evgen/Downloads/s_1_1102_c.tif")
    imagebefore=image
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    boxplot_2d(image[0:int(image.shape[0]),:],image[:,0:int(image.shape[1])],ax=ax2, whis=7)
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    image=filtration(image)
    clusters(image)
    plt.show()
    
    
if __name__ == "__main__":
    main()  
    
    
        
    


# In[ ]:




