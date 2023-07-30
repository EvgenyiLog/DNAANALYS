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

from sklearn.decomposition import FastICA,PCA,KernelPCA
def compresed(hlpfilt):
    relative_rank = 0.9
    max_rank = int(relative_rank * min(hlpfilt.shape[0], hlpfilt.shape[1]))
    print("max rank = %d" % max_rank)
    U,S,VT=np.linalg.svd(hlpfilt)
    A = np.zeros((U.shape[0], VT.shape[1]))
    k=max_rank//2
    for i in range(1,int(k)):
        U_i = U[:,[i]]
        VT_i = np.array([VT[i]])
        A += S[i] * (U_i @ VT_i)
    compressed_float=(U[:,:k] @ np.diag(S[:k])) @ VT[:k]
    compressed = (np.minimum(compressed_float, 1.0) * 0xff).astype(np.uint8)
    compreseds=cv2.cvtColor(compressed, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_compresedsvd.jpg",compreseds)
    plt.figure(figsize=(15,7))
    plt.imshow(compressed[0:1000,0:1000],cmap='gray',vmax=compressed.max(),vmin=compressed.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    pca = PCA(n_components=hlpfilt.shape[0]//2)
    hlpfilt_new=pca.fit_transform(hlpfilt)
    compresed=pca.inverse_transform(hlpfilt_new)
    compreseds=cv2.cvtColor(compressed, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_compresedpca.jpg",compreseds)
    plt.figure(figsize=(15,7))
    plt.imshow(compressed[0:1000,0:1000],cmap='gray',vmax=compressed.max(),vmin=compressed.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    fastica = FastICA(n_components=hlpfilt.shape[0]-1)
    hlpfilt_new=fastica.fit_transform(hlpfilt)
    compresed=fastica.inverse_transform(hlpfilt_new)
    compreseds=cv2.cvtColor(compressed, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_compresedfastica.jpg",compreseds)
    plt.figure(figsize=(15,7))
    plt.imshow(compresed[0:1000,0:1000],cmap='gray',vmax=compressed.max(),vmin=compressed.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    try:
        transformer = KernelPCA(n_components=hlpfilt.shape[0]//2, kernel='rbf')
        hlpfilt_new=transformer.fit_transform(hlpfilt)
        compresed=transformer.inverse_transform(hlpfilt_new)
        compreseds=cv2.cvtColor(compressed, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_compresedkernelpca.jpg",compreseds)
        plt.figure(figsize=(15,7))
        plt.imshow(compresed[0:1000,0:1000],cmap='gray',vmax=compressed.max(),vmin=compressed.min())
        #plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    except:
        pass

            
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def normcorr(image1,image2):
    d = 1

    correlation = np.zeros_like(image1)
    sh_row, sh_col = image1.shape
    for i in range(d, sh_row - (d + 1)):
        for j in range(d, sh_col - (d + 1)):
            correlation[i, j] = correlation_coefficient(image1[i - d: i + d + 1,j - d: j + d + 1],image2[i - d: i + d + 1,j - d: j + d + 1])

    correlation=np.asarray(correlation,dtype=np.uint8)
    plt.figure(figsize=(15,7))
    plt.imshow(correlation[0:1000,0:1000],cmap='gray',vmax=correlation.max(),vmin=correlation.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    try:
        r=cv2.cvtColor(correlation, cv2.COLOR_GRAY2BGR)
    except:
        r=correlation
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_normxcorr.jpg",r)     

def imageground(image):
    # Вычисляем маску фона
    image=np.uint8(image)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Убираем шум
    kernel = np.ones((2, 2), np.uint16)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=50)
    background=opening
    backgrounds=cv2.cvtColor( background, cv2.COLOR_GRAY2BGR) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_background1.jpg",backgrounds)
    plt.figure(figsize=(15, 7))
    plt.imshow(background[0:1000,0:1000], cmap=plt.cm.gray,vmax=background.max(),vmin=background.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    # создаем маску фона
    sure_bg = cv2.dilate(opening, kernel, iterations=20) 


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  

    foreground=sure_fg
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_foreground.jpg",foreground)
    plt.figure(figsize=(15, 7))
    plt.imshow(foreground[0:1000,0:1000], cmap=plt.cm.gray,vmax=foreground.max(),vmin=foreground.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    

    # Вычисляем "неопределенный регион" из вычисленных фона и переднего плана
    # В большинстве случаев это просто фон, т.к. передний план считается плохо.
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Размечаем маркеры. Они будут использованы для вычисления фона алгоритмом watershed
    _, markers = cv2.connectedComponents(sure_fg)
    # Добавляем единичку ко всем значениям, чтобы они были больше 0. Он отвечает за "неизвестную" область.
    markers = markers + 1
    # Добавляем "неизвестную" область на маркеры
    #markers[unknown == 255] = 0

    # собственно считаем
    #markers = cv2.watershed(image, markers)
    # и рисуем наши маркеры на изображении
    #img[markers == -1] = [255, 0, 0]


def process(img):
    work_img = img.copy()

    # считаем маску
    _, threshold = cv2.threshold(work_img, 30, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ищем на маске контуры
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = []

    # ищем самый большой из найденных контуров
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

        cnt_area = cv2.contourArea(cnt)
        # обязательно проверяем кол-во точек.
        # Иначе, часто, получается что самая большая фигура -
        # это треугольник через всю диагональ изображения
        if len(approx) > 3 and cnt_area > max_area:
            max_area = cnt_area
            max_contour = approx

    # рисуем самый большой контур на изображении
    cv2.drawContours(work_img, [max_contour], 0, (255, 0, 0), 2)

    return work_img, threshold            
                     
def corrfft(image1,image2):
    image1=np.uint8(image1)
    image2=np.uint8(image2)
    dft1 = cv2.dft(np.float32(image1),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(image2),flags = cv2.DFT_COMPLEX_OUTPUT)
    corr=cv2.multiply(dft1,dft2.conj())
    corr/=np.amax(corr)
    #plt.figure(figsize=(15,7))
    #plt.imshow(corr[0:1000,0:1000],cmap='gray',vmax=corr.max(),vmin=corr.min())
    #plt.grid(True)
    #plt.tick_params(labelsize =20,#  Размер подписи
                    #color = 'k')   #  Цвет делений
        
        
        
def findcountour(image):
    imagesource=image
    'поиск контуров'
    edges = cv2.Canny(image=image, threshold1=1, threshold2=2)
   
    plt.figure(figsize=(15,7))
    plt.imshow(edges[0:1000,0:1000],cmap='gray',vmax=edges.max(),vmin=edges.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    xcentr=[]
    ycentr=[]
    xcentrrect=[]
    ycentrrect=[]
    intensivity=[]
    intensivityrect=[]
    ret, thresh = cv2.threshold(edges, 1, 2, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    
    for i in contours:
        M = cv2.moments(i)
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        #print(x)
        xcentrrect.append(x)
        ycentrrect.append(y)
        intensivityrect.append(imagesource[int(np.floor(y)),int(np.ceil(x))])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr.append(cx)
            ycentr.append(cy)
            intensivity.append(imagesource[cy,cx])
            
            
    d={'xcentr':xcentr,'ycentr':ycentr,'intensivity':intensivity}
    df=pd.DataFrame(data=d)
    df.to_excel("C:/Users/evgen/Downloads/contourcentrintensivity.xlsx")
    d={'xcentrrect':xcentrrect,'ycentrrect':ycentrrect,'intensivityrect':intensivityrect}
    df=pd.DataFrame(data=d)
    df.to_excel("C:/Users/evgen/Downloads/contourcentrintensivityrect.xlsx")
            
            
def main():
    image=tiffreader("C:/Users/evgen/Downloads/s_1_1102_c.tif")
    image=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
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
    print(image.shape)
    plt.figure(figsize=(15,7))
    plt.imshow(image[0:1000,0:1000],cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    imageplot3d(image)
    localstdmean(image,2)
    findcountour(image)
    
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




