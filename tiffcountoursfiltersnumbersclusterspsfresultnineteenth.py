#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    
    
def countourfind(image):
    imagesource=image
    #print(imagesource.shape)
    'поиск контуров'
    #image=cv2.cvtColor(image, cv2.CV_16U)
    #print(image.shape)
    edges = cv2.Canny(np.uint8(image), threshold1=1, threshold2=2)
   
    plt.figure(figsize=(15,7))
    plt.imshow(edges[0:1000,0:1000],cmap='gray',vmax=edges.max(),vmin=edges.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    ret, thresh1 = cv2.threshold(edges, 1, 2, 0)
    plt.figure(figsize=(15,7))
    plt.imshow(thresh1[0:1000,0:1000],cmap='gray',vmax=thresh1.max(),vmin=thresh1.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    image1=cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_thresh1.jpg",image1)
    thresh2=cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 3, 1)
    plt.figure(figsize=(15,7))
    plt.imshow(thresh2[0:1000,0:1000],cmap='gray',vmax=thresh2.max(),vmin=thresh2.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    image2=cv2.cvtColor(thresh2,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_thresh2.jpg",image2)
    
    
    ret, thresh3 = cv2.threshold(image, 1, 2, 0)
    plt.figure(figsize=(15,7))
    plt.imshow(thresh3[0:1000,0:1000],cmap='gray',vmax=thresh3.max(),vmin=thresh3.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.figure(figsize=(15,7))
    plt.imshow(thresh3[0:1000,0:1000],cmap='gray',vmax=thresh3.max(),vmin=thresh3.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    image3=cv2.cvtColor(thresh3,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_thresh3.jpg",image3)
   
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    xcentr=[]
    ycentr=[]
    xcentrrect=[]
    ycentrrect=[]
    areas=[]
    perimeters = []
    mask=np.zeros_like(imagesource,dtype=np.uint16)
    #print(mask.shape)
    sumpixel=[]
    #print(image.shape)
    image= cv2.cvtColor(imagesource,cv2.COLOR_GRAY2RGB)
    
    #print(image.dtype)
    #print(image.shape)
    hsl=cv2.cvtColor(np.uint8(image),cv2.COLOR_RGB2HLS)
    h1,s1,l1=cv2.split(hsl)
    hsv=cv2.cvtColor(np.uint8(image),cv2.COLOR_RGB2HSV)
    h2,s2,v=cv2.split(hsv)
    lab=cv2.cvtColor(np.uint8(image),cv2.COLOR_RGB2LAB)
    l3,a,b=cv2.split(lab)
    lcenters1=[]
    hcenters1=[]
    vcenters=[]
    scenters1=[]
    lcenters2=[]
    hcenters2=[]
    lcenters3=[]
    acenters=[]
    bcenters=[]
    sumpixelrect=[]
    sumpixel=[]

    
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr.append(cx)
            ycentr.append(cy)
            lcenters1.append(l1[cy,cx])
            #lcenters2.append(l2[cy,cx])
            #lcenters3.append(l3[cy,cx])
            hcenters1.append(h1[cy,cx])
            #hcenters2.append(h2[cy,cx])
            #hcenters2.append(h2[cy,cx])
            acenters.append(a[cy,cx])
            bcenters.append(b[cy,cx])
            vcenters.append(v[cy,cx])
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        rect=cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        xcentrrect.append(x)
        ycentrrect.append(y)
        areas.append(cv2.contourArea(i))
        perimeters.append(cv2.arcLength(i,True))
        imagefill=cv2.fillPoly(mask, i, 1)
        #print(imagefill.shape)
        masked_image = cv2.bitwise_and(imagefill,imagesource)
        sumpixel.append(np.sum(masked_image))
        box = np.int0(box)
        masked_imagerect=cv2.bitwise_and(cv2.rectangle(mask,box[2],box[3],1, 1),imagesource)
        sumpixelrect.append(np.sum(masked_imagerect))
        #(x,y,w,h) = cv2.boundingRect(i)
        #cv2.rectangle(imagesource, (x,y), (x+w,y+h), (0,255,0), 2)
        
        
    print('Количество')
    print(len(xcentr))
    print()   
        
    d={'xcentr':xcentr,'ycentr':ycentr,'areas': areas,'sumpixel':sumpixel,'bcenters':bcenters,'acenters':acenters,
       'sumpixelrect': sumpixelrect,'perimeters':perimeters,'xcentrrect':xcentrrect,'ycentrrect':ycentrrect,
       'lightness ':lcenters1,'value ': vcenters,'Saturation':scenters1,'hue':hcenters1}
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contour.csv")
    df.to_excel("C:/Users/evgen/Downloads/contour.xlsx")    
                                   
                               

            
            
        
    
    
def main():
    image=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    image=tiffreader("C:/Users/evgen/Downloads/s_1_1102_c.tif")
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    boxplot_2d(image[0:int(image.shape[0]),:],image[:,0:int(image.shape[1])],ax=ax2, whis=1.5)
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    #image=readimage("C:/s_1_1102_c.jpg")
    #image=readimage("C:/s_1_1101_a.jpg")
    #image=image[0:1000,0:1000]
    #print(image.shape)
    plt.figure(figsize=(15,7))
    plt.imshow(image[0:1000,0:1000],cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    imageplot3d(image)
    countourfind(image)
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




