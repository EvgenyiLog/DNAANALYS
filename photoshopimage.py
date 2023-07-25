#!/usr/bin/env python
# coding: utf-8

# In[8]:


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
from skimage import io, segmentation, color
from skimage.future import graph

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
    
    image = cv2.imread(path,-1)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    print(image.shape)
    print(image.dtype)
    print(f'image max={np.amax(image)}')
    print(f'image min={np.amin(image)}')
    print(f'image mean={np.mean(image)}')
    print(f'image std={np.std(image)}')
    return image

def tiffreader(path):
    image=cv2.imread(path,-1)
    
    #(image.shape)
    print(image.dtype)
    return image


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


def binaryimage(image):
    imagepsnr=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    entr_img = entropy(imagepsnr, disk(1))
    plt.figure(figsize=(15,7))
    plt.imshow(entr_img,cmap='gray',vmax=entr_img.max(),vmin=entr_img.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    psnr=peak_signal_noise_ratio(imagepsnr, std_convoluted(imagepsnr,2))
    print(f'PSNR before filtration={psnr}')
    print(f'PSNR before filtration={20*np.log10(1/imagepsnr.std())}')
   
    try:
        mse = mean_squared_error(imagepsnr,imagepsnr.T)
        print(f'PSNR before filtration={20*np.log10(1/mse)}')
    except:
        pass
    try:
         psnr=cv2.PSNR(image, image.T)
         print(f'PSNR before filtration={psnr}')
    except:
        pass
    
    
     
    print(f'PSNR before filtration max std={20*np.log10(imagepsnr.max()/imagepsnr.std())}')

    
    image=scipy.signal.wiener(image,mysize=1,noise=np.std(image))
    image=np.asarray(image,dtype=np.uint8)
    image=cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    #print(image.shape)
    #print(image.dtype)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1., tileGridSize=(2,2))
    l, a, b = cv2.split(image)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    #image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image=cv2.bilateralFilter(image,1,1,1)
    #print(image.shape)
    #image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    #print(image.shape)
    image=adjust_gamma(image,1.0)
    imagegray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    slp,hlp=sporco.signal.tikhonov_filter(imagegray,3)
    fig=plt.figure(figsize=(15, 7))
    plt.imshow(slp, cmap=plt.cm.gray,vmax=slp.max(),vmin=slp.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    fig=plt.figure(figsize=(15, 7))
    plt.imshow(hlp, cmap=plt.cm.gray,vmax=hlp.max(),vmin=hlp.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    images=cv2.cvtColor(slp,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexslp.jpg",images)
    images=cv2.cvtColor(hlp,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexhlp.jpg",images)
    print(image.shape)
    imagepsnr=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    entr_img = entropy(imagepsnr, disk(1))
    plt.figure(figsize=(15,7))
    plt.imshow(entr_img,cmap='gray',vmax=entr_img.max(),vmin=entr_img.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    psnr=peak_signal_noise_ratio(imagepsnr, std_convoluted(imagepsnr,2))
    print(f'PSNR after filtration={psnr}')
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
    plt.figure(figsize=(15, 7))
    plt.imshow(image,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    images=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alex.jpg",images)
    
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    boxplot_2d(image[0:int(image.shape[0]),:],image[:,0:int(image.shape[1])],ax=ax2, whis=1.5)
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    plt.figure(figsize=(15, 7))
    plt.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    images=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexgray.jpg",images)
    
    #thresh = np.std(image)
    thresh = np.mean(image)
    #thresh =128
    imagebinary = cv2.threshold(image, thresh, np.amax(image), cv2.THRESH_BINARY)[1]
    fig=plt.figure(figsize=(15, 7))
    plt.imshow(imagebinary, cmap=plt.cm.gray,vmax=imagebinary.max(),vmin=imagebinary.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    images=cv2.cvtColor(imagebinary,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexbinary.jpg",images)
    
    # threshold input image using otsu thresholding as mask and refine with morphology
    ret, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # put mask into alpha channel of result
    result = image.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    fig=plt.figure(figsize=(15, 7))
    plt.imshow(result, cmap=plt.cm.gray,vmax=result.max(),vmin=result.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    cv2.imwrite("C:/Users/evgen/Downloads/alexresult.jpg",result)
    
    labels = segmentation.slic(image, compactness=30, n_segments=900, start_label=1)
    g = graph.rag_mean_color(image, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=20, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

    out = color.label2rgb(labels2, image, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    
    plt.figure(figsize=(15,7))
    plt.imshow(out,cmap='gray',vmax=out.max(),vmin=out.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    out=np.asarray(out,dtype=np.uint8)
    r=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexRAG.jpg",r)
    
    edges = cv2.Canny(image=image, threshold1=15, threshold2=30)
   
    plt.figure(figsize=(15,7))
    plt.imshow(edges,cmap='gray',vmax=edges.max(),vmin=edges.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    images=cv2.cvtColor(edges,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexedges.jpg",images)
    
    
    ret, thresh = cv2.threshold(edges, 15, 30, 0)
    plt.figure(figsize=(15,7))
    plt.imshow(thresh,cmap='gray',vmax=thresh.max(),vmin=thresh.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    r=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexthresh.jpg",r)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    xcentr=[]
    ycentr=[]
    xcentrrect=[]
    ycentrrect=[]
    areas=[]
    perimeters = []
    intensivity=[]
    imagergb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr.append(cx)
            ycentr.append(cy)
            intensivity.append(imagergb[cy,cx])
        r=cv2.fillPoly(imagergb, i, color=(255, 0, 0))
        
            
    r=cv2.cvtColor(r,cv2.COLOR_BGR2RGB)  
    cv2.imwrite("C:/Users/evgen/Downloads/alexcountour.jpg",r)
    
    imagergb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    for i in contours:
       
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        xcentrrect.append(x)
        ycentrrect.append(y)
        rect=cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
       
        box = np.int0(box)
        
        rect=cv2.rectangle(imagergb,box[2],box[3],(0,255,0), 2)
    rect=cv2.cvtColor(rect,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexcountourrect.jpg",rect)
    
    
    imagergb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    
    for i in contours:
        M = cv2.moments(i)
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            circlecountourcentr=cv2.circle(imagergb, (int(cx),int(cy)), 3, (255,0,0), 2)
        
    r=cv2.cvtColor(circlecountourcentr,cv2.COLOR_BGR2RGB)  
    cv2.imwrite("C:/Users/evgen/Downloads/alexcirclecountourcentr.jpg",r)
    imagergb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    for i in contours:
        M = cv2.moments(i)
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        circlerectcentr=cv2.circle(imagergb, (int(x),int(y)), 3, (0,255,0), 2)
        
   
    rect=cv2.cvtColor(circlerectcentr,cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/alexcirclerectcentr.jpg",rect)
    
    
    imagergb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    for i in contours:
        M = cv2.moments(i)
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            circlecountourcentr1=cv2.circle(imagergb, (int(cx),int(cy)), 3, (255,0,0), 2)
        circlerectcentr1=cv2.circle(imagergb, (int(x),int(y)), 3, (0,255,0), 2)
        
        
    r=cv2.cvtColor(cv2.add(circlecountourcentr1,circlerectcentr1),cv2.COLOR_BGR2RGB)  
    cv2.imwrite("C:/Users/evgen/Downloads/alexcirclecountourcentrrectcent.jpg",r)
    
    
    
    
def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-07-21_14-30-42.jpg")
    #image=readimage("C:/Users/evgen/Downloads/photo_2023-06-24_22-06-01.jpg")
    #image=readimage("C:/Users/evgen/Downloads/photo_2023-06-24_22-07-55.jpg")
    #image=readimage("C:/Users/evgen/Downloads/photo_2023-06-18_22-11-39.jpg")
    #image=readimage("C:/Users/evgen/Documents/My Snips/capture20230723140421973.png")
    plt.figure(figsize=(15, 7))
    plt.imshow(image,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    boxplot_2d(image[0:int(image.shape[0]),:],image[:,0:int(image.shape[1])],ax=ax2, whis=1.5)
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    binaryimage(image)
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




