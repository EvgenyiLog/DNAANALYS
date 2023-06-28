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
    

from sklearn.decomposition import FastICA,PCA
from skimage.feature import hog
def filtration(image,path):
    'path to file correlate'
    'image filtration'
    'фильтрация возвращает фильтрованное изображение'
    imagepsnr=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    entr_img = entropy(imagepsnr, disk(1))
    plt.figure(figsize=(15,7))
    plt.imshow(entr_img[0:1000,0:1000],cmap='gray',vmax=entr_img.max(),vmin=entr_img.min())
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

    
    image=scipy.signal.wiener(image,noise=image.std())
    
    
    image=np.asarray(image,dtype=np.uint8)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image= cv2.bilateralFilter(image,1,1,1)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_wiener.jpg",image)
    image=cv2.fastNlMeansDenoising(image,None,1,1,3)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_fastNlMeansDenoising.jpg",image)
    #image=cv2.fastNlMeansDenoising(image,None,1,1,3)
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    clahe = cv2.createCLAHE(clipLimit=5., tileGridSize=(2,2))
    l, a, b = cv2.split(image)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    slp,hlp=sporco.signal.tikhonov_filter(image,3)
    hlp=ndimage.median_filter(hlp, size=11)
    hlp=hlp<np.percentile(hlp, 90)
    image2 = readimage(path)
    #image2=image2[0:1000,0:1000]
    image2=np.asarray(image2,dtype=np.uint8)

    mse = mean_squared_error(hlp, image2)
    ssim = structural_similarity(hlp, image2, data_range=image2.max() - image2.min())
    print('mse')
    print(mse)
    print('ssim')
    print(ssim)
    print()
    print(hlp.shape)
    print(image2.shape)
    print()
    image2=np.asarray(image2,dtype=np.uint8)
    
    #normcorr(hlp,image2)
    res = cv2.matchTemplate(np.uint8(hlp),image2,cv2.TM_CCORR_NORMED)
    r=cv2.cvtColor(res, cv2.COLOR_GRAY2RGB) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_normcorr2.jpg",r)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    eng = matlab.engine.start_matlab()   
    r=eng.corr2(hlp,image2)
    
    hlp=hlp-hlp.mean()-r*image2
    try:
        hlpf=hlp-hlp.mean()-cv2.multiply(res,image2)
        hlpfilt=np.asarray(hlpf,dtype=np.uint8)
        hlpfilt=cv2.cvtColor(hlpfilt,cv2.COLOR_GRAY2RGB)
        cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_cfilt1.jpg",hlpfilt)
    except:
        pass
    
    
    hlpfilt=np.asarray(hlp,dtype=np.uint16)
    #hlpfilt = eng.imadjust(hlpfilt,eng.stretchlim(hlpfilt),[])
    hlpfilt=eng.imadjust(hlpfilt)
    hlpfilt=np.asarray(hlpfilt,dtype=np.uint8)
    hlpfilt=cv2.cvtColor(hlpfilt,cv2.COLOR_GRAY2RGB)
    #hlp=scipy.ndimage.percentile_filter(hlp,95)
    
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_cfilt.jpg",hlpfilt)
    #cv2.imwrite("C:/Users/Евгений/Downloads/s_1_1102_cfilt.jpg",hlpfilt)
    normcorr(hlp[0:100,0:100],image2[0:100,0:100])
    corrfft(hlp,image2)
    r=eng.normxcorr2(hlp,image2)
    r=np.asarray(r,dtype=np.uint8)
    print(r.shape)
    
    r=cv2.cvtColor(r, cv2.COLOR_GRAY2RGB) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_normxcorr2.jpg",r)
    #cv2.imwrite("C:/Users/Евгений/Downloads/s_1_1102_c_a_normxcorr2.jpg",r)
    eng.quit()
    hlpfilt=cv2.cvtColor(hlpfilt,cv2.COLOR_RGB2GRAY)
    imageh=hessian(hlpfilt)
    plt.figure(figsize=(15,7))
    plt.imshow(imageh[0:1000,0:1000],cmap='gray',vmax=imageh.max(),vmin=imageh.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    r=cv2.cvtColor(np.uint8(imageh), cv2.COLOR_GRAY2RGB) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_hessian.jpg",r)
    try:
        im=cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2RGB) 
        fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_hog.jpg",hog_image)
    except:
        pass
    sigma_est =skimage.restoration.estimate_sigma(hlpfilt)
    imagew=skimage.restoration.denoise_wavelet(hlpfilt,sigma=sigma_est)
    plt.figure(figsize=(15,7))
    plt.imshow(imagew[0:1000,0:1000],cmap='gray',vmax=imagew.max(),vmin=imagew.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    psnr=peak_signal_noise_ratio(hlpfilt, std_convoluted(hlpfilt,2))
    print(f'PSNR after filtration={psnr}')
    print(f'PSNR after filtration={20*np.log10(1/hlpfilt.std())}')
    try:
        mse = mean_squared_error(hlpfilt,hlpfilt.T)
        print(f'PSNR after filtration={20*np.log10(1/mse)}')
    except:
        pass
    try:
        psnr=cv2.PSNR(hlpfilt,hlpfilt.T)
        print(f'PSNR after filtration={psnr}')
    except:
        pass
    
    print(f'PSNR after filtration max std={20*np.log10(hlpfilt.max()/hlpfilt.std())}')
    
    
    
    
    
    
    entr_img = entropy(hlpfilt, disk(1))
    plt.figure(figsize=(15,7))
    plt.imshow(hlpfilt[0:1000,0:1000],cmap='gray',vmax=hlpfilt.max(),vmin=hlpfilt.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    return hlpfilt


from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
def detect_ridges(gray, sigma=1.0):
    #hxx, hyy, hxy = hessian_matrix(gray, sigma)
    H_elems = hessian_matrix(gray, sigma)
    #i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    i1=hessian_matrix_eigvals(H_elems)[0]
    i2=hessian_matrix_eigvals(H_elems)[1]
    return i1, i2


def ridgedetect(image):
    imagemax,imagemin=detect_ridges(image, sigma=1.0)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagemax[0:1000,0:1000], cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    plt.figure(figsize=(15, 7))
    plt.imshow(imagemin[0:1000,0:1000], cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    try:
       ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
       ridges = ridge_filter.getRidgeFilteredImage(image)
       plt.figure(figsize=(15, 7))
       plt.imshow(ridges[0:1000,0:1000], cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
       plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    except:
        pass
    

def segmentationimage(edges):
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    r=cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Build model
    model = get_model(r.size, 3)
    model.summary()
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
    keras.callbacks.ModelCheckpoint("C:/Users/evgen/Downloads/segmentation.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 5
    model.fit(r, epochs=epochs, validation_data=r, callbacks=callbacks)
    val_preds = model.predict(r)
    plt.figure(figsize=(15,7))
    plt.imshow(val_preds[0:1000,0:1000],cmap='gray',vmax=val_preds.max(),vmin=val_preds.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    mask = np.argmax(val_preds, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
    plt.figure(figsize=(15,7))
    plt.imshow(img[0:1000,0:1000],cmap='gray',vmax=img.max(),vmin=img.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    


from scipy import special
import pylab
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy import ndimage as ndi
from skimage import filters
from scipy import ndimage
from skimage.measure import find_contours

def findcountur(image):
    contours = measure.find_contours(image, 0.8)
    plt.figure(figsize=(15, 7))
    plt.imshow(image[0:1000,0:1000], cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    plt.figure(figsize=(15, 7))
    for contour in contours:
        plt.plot(contour[0:1000, 1], contour[0:1000, 0], linewidth=2)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        
    light_white =(255, 255, 247)
    dark_white = (225, 217, 209)
    imagergb= cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    mask_white = cv2.inRange(imagergb, light_white, dark_white)
    result = cv2.bitwise_and(imagergb, imagergb, mask=mask_white)
    result= cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
    plt.figure(figsize=(15, 7))
    plt.imshow(result[0:1000,0:1000],cmap='gray',vmax=result.max(),vmin=result.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    coordinates = peak_local_max(image, min_distance=2)
    xcentrmax=coordinates[:,1]
    ycentrmax=coordinates[:,0]
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    print('Количество локальных максимумов')
    print(len(xcentrmax))


    thresholds = filters.threshold_multiotsu(image, classes=3)
    regions = np.digitize(image, bins=thresholds)

    plt.figure(figsize=(15, 7))
    plt.imshow(regions[0:1000,0:1000],cmap='gray',vmax=regions.max(),vmin=regions.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    localmax=ndimage.maximum_filter(image, size=2)
    plt.figure(figsize=(15, 7))
    plt.imshow(localmax[0:1000,0:1000],cmap='gray',vmax=localmax.max(),vmin=localmax.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    localmin=ndimage.minimum_filter(image, size=2)
    plt.figure(figsize=(15, 7))
    plt.imshow(localmin[0:1000,0:1000],cmap='gray',vmax=localmin.max(),vmin=localmin.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    thresholds = filters.threshold_multiotsu(image, classes=3)
    regions = np.digitize(image, bins=thresholds)

    plt.figure(figsize=(15, 7))
    plt.imshow(regions[0:1000,0:1000],cmap='gray',vmax=regions.max(),vmin=regions.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    localmax=ndimage.maximum_filter(image, size=2)
    plt.figure(figsize=(15, 7))
    plt.imshow(localmax[0:1000,0:1000],cmap='gray',vmax=localmax.max(),vmin=localmax.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    localmin=ndimage.minimum_filter(image, size=2)
    plt.figure(figsize=(15, 7))
    plt.imshow(localmin[0:1000,0:1000],cmap='gray',vmax=localmin.max(),vmin=localmin.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений


def countourfind(image):
    imagesource=image
    'поиск контуров'
    edges = cv2.Canny(image=image, threshold1=1, threshold2=2)
   
    plt.figure(figsize=(15,7))
    plt.imshow(edges[0:1000,0:1000],cmap='gray',vmax=edges.max(),vmin=edges.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    x_train=np.asarray(edges,dtype=float)
       
       
    encoding_dim =8192
       
    input = keras.Input(shape=(image.shape[1],))
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(input)
    decoded = keras.layers.Dense(image.shape[1], activation='relu')(encoded)
       
    # his model maps an input to its reconstruction
    autoencoder = keras.Model(input, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='mse')
    autoencoder.fit(x_train, x_train,epochs=20,batch_size=512)
    edges=autoencoder.predict(x_train)
    edges=np.asarray(edges,dtype=np.uint8)
    plt.figure(figsize=(15,7))
    plt.imshow(edges[0:1000,0:1000],cmap='gray',vmax=edges.max(),vmin=edges.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    r=cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_edges.jpg",r)
    #cv2.imwrite("C:/Users/Евгений/Downloads/s_1_1102_c_edges.jpg",r)
    ret, thresh = cv2.threshold(edges, 1, 2, 0)
    r=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_countour_thresh.jpg",r)
    thresh1=cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 3, 1)
    r=cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_countour_threshadapt.jpg",r)
    contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    xcentra=[]
    ycentra=[]
    areasa=[]
    perimetersa = []
    image=np.zeros((imagesource.shape[0],imagesource.shape[1],3),dtype=np.uint8)
    for i in contours1:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentra.append(cx)
            ycentra.append(cy)
        areasa.append(cv2.contourArea(i))
        perimetersa.append(cv2.arcLength(i,True))
        r=cv2.fillPoly(image, i, color=(255, 255, 255))
        
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_countour_adapt.jpg",r)   

    print('Количество')
    print(len(xcentra))
    print()
    d={'xcentr':xcentra,'ycentr':ycentra}
     
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contourafterfiltrationadapt.csv")
    df.to_excel("C:/Users/evgen/Downloads/contourafterfiltrationadapt.xlsx")
    d={'areas': areasa,'perimeters':perimetersa}
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contourafterfiltrationadaptarp.csv")
    df.to_excel("C:/Users/evgen/Downloads/contourafterfiltrationadaptarp.xlsx")

    r=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_countour_thresh.jpg",r)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    xcentr=[]
    ycentr=[]
    xcentrrect=[]
    ycentrrect=[]
    areas=[]
    perimeters = []
    mask=np.zeros_like(imagesource,dtype=np.uint8)
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
    meanpixel=[]

    
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr.append(cx)
            ycentr.append(cy)
        r=cv2.fillPoly(image, i, color=(255, 255, 255))
            
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        rect=cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        xcentrrect.append(x)
        ycentrrect.append(y)
       
        imagefill=cv2.fillPoly(mask, i, 1)
        #print(imagefill.shape)
        masked_image = cv2.bitwise_and(imagefill,imagesource,mask)
        meanpixel.append(cv2.mean(masked_image))
        #m,s=cv2.meanStdDev(masked_image)
        sumpixel.append(np.sum(masked_image))
        box = np.int0(box)
        masked_imagerect=cv2.bitwise_and(cv2.rectangle(mask,box[2],box[3],1, 1),imagesource,mask)
        sumpixelrect.append(np.sum(masked_imagerect))
        
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_countour_adapt.jpg",r)   
        
        
    print('Количество')
    print(len(xcentr))
    print()
        
        
    d={'xcentr':xcentr,'ycentr':ycentr}
       
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contourafterfiltration.csv")
    df.to_excel("C:/Users/evgen/Downloads/contourafterfiltration.xlsx") 
    
    
    for i in contours:
        M = cv2.moments(i)
        areas.append(cv2.contourArea(i))
        perimeters.append(cv2.arcLength(i,True))
        
    
    d={'areas': areas,'perimeters':perimeters}
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contourafterfiltrationarp.csv")
    df.to_excel("C:/Users/evgen/Downloads/contourafterfiltrationarp.xlsx") 
    
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            lcenters1.append(l1[cy,cx])
            #lcenters2.append(l2[cy,cx])
            #lcenters3.append(l3[cy,cx])
            hcenters1.append(h1[cy,cx])
            #hcenters2.append(h2[cy,cx])
            #hcenters2.append(h2[cy,cx])
            acenters.append(a[cy,cx])
            bcenters.append(b[cy,cx])
            vcenters.append(v[cy,cx])
    
    d={'lightness ':lcenters1,'value ': vcenters,'Saturation':scenters1,'hue':hcenters1,'bcenters':bcenters,'acenters':acenters}
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contourafterfiltrationhsvl.csv")
    df.to_excel("C:/Users/evgen/Downloads/contourafterfiltrationhvsl.xlsx") 
    
    for i in contours:
        M = cv2.moments(i)
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        rect=cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        xcentrrect.append(x)
        ycentrrect.append(y)
        imagefill=cv2.fillPoly(mask, i, 1)
        #print(imagefill.shape)
        masked_image = cv2.bitwise_and(imagefill,imagesource,mask)
        meanpixel.append(cv2.mean(masked_image))
        #m,s=cv2.meanStdDev(masked_image)
        sumpixel.append(np.sum(masked_image))
        box = np.int0(box)
        masked_imagerect=cv2.bitwise_and(cv2.rectangle(mask,box[2],box[3],1, 1),imagesource,mask)
        sumpixelrect.append(np.sum(masked_imagerect))
    
    d={'xcentrrect':xcentrrect,'ycentrrect':ycentrrect}
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contourafterfiltrationrect.csv")
    df.to_excel("C:/Users/evgen/Downloads/contourafterfiltrationrect.xlsx") 
    
    d={'meanpixel': meanpixel,'sumpixel':sumpixel,'sumpixelrect': sumpixelrect}
    df=pd.DataFrame(data=d)
    
    
    
    df.to_csv("C:/Users/evgen/Downloads/contourafterfiltrationsumpixel.csv")
    df.to_excel("C:/Users/evgen/Downloads/contourafterfiltrationsumpixel.xlsx")
    
    
    plt.figure(figsize =(15, 7))
    plt.boxplot(sumpixelrect)
    plt.grid(True)
    #plt.ylim(0,100)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    plt.figure(figsize=(15,7))
    plt.hist(xcentr,bins=50,density=True,stacked=True, facecolor='r',histtype= 'bar',edgecolor='k',linewidth=2, alpha=0.75)
    plt.grid(True)
    #plt.ylim(0,100)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    plt.figure(figsize=(15,7))
    plt.hist(ycentr,bins=50,density=True,stacked=True, facecolor='r',histtype= 'bar',edgecolor='k',linewidth=2, alpha=0.75)
    plt.grid(True)
    #plt.ylim(0,100)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    plt.figure(figsize=(15,7))
    plt.hist2d(xcentr, ycentr,bins = 10, cmap ="gray")
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
 
    isClosed = True
    image3=np.zeros((image.shape[0],image.shape[1],3))

    color = (255, 255, 255)
    image=np.asarray(image,dtype=np.uint8)
    # Line thickness of 5 px
    thickness = 1
 
    # Using cv2.polylines() method


    image4 = cv2.polylines(image3, contours,
                      isClosed, color, thickness)
    plt.figure(figsize=(15,7))
    plt.imshow(image4[0:1000,0:1000],cmap='gray',vmax=image4.max(),vmin=image4.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_contours.jpg",image4)
    for i in range(len(xcentr)):
        image5=cv2.circle(image3, (xcentr[i], ycentr[i]), 1, (255, 255, 255), -1)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_centers.jpg",image5)
    plt.figure(figsize=(15,7))
    plt.imshow(image5[0:1000,0:1000],cmap='gray',vmax=image5.max(),vmin=image5.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    image6=cv2.fillPoly(image3,contours,color)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_cluster.jpg",image6)
    plt.figure(figsize=(15,7))
    plt.imshow(image6[0:1000,0:1000],cmap='gray',vmax=image6.max(),vmin=image6.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    image= cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    hsl=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    h,s,l=cv2.split(hsl)
    
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    
    
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(xcentr, ycentr, bins=(50,50))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    dz=dz/dz.sum()

    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    
   
    
    
    
   
    
    
    
    
    
    

    
    eng = matlab.engine.start_matlab()
    try:
        image=eng.imcontour(imagesource,1)
    except:
        pass
    eng.quit()
    print('Quality')
    xcentr=np.asarray(xcentr,dtype=float)
    print(-10*np.log10(special.erfc(xcentr.mean()/xcentr.std())))
    print(-10*np.log10(special.erfc(len(xcentr))))
    print()
    
    session=WolframLanguageSession("C:/Program Files/Wolfram Research/Mathematica/12.1/WolframKernel.exe")
    EdgeDetect=session.function(wl.EdgeDetect)
    try:
        image=image[0:500,0:500]
        imageedges=EdgeDetect(image,1)
        cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_edgeswolfram.jpg",imageedges)
    except:
        pass
    return edges


def findcounturnewmethod(edges):
   
    ret, thresh = cv2.threshold(edges, 1, 2, 0)
    contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    xcentr1=[]
    ycentr1=[]
    for i in contours1:
        M = cv2.moments(i)
        if M['m00'] != 0:
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr1.append(cx)
            ycentr1.append(cy)

    d={'xcentr1':xcentr1,'ycentr1':ycentr1}
    df=pd.DataFrame(data=d)
    
    df.to_csv("C:/Users/evgen/Downloads/contour1.csv")
    df.to_excel("C:/Users/evgen/Downloads/contour1.xlsx") 


  


    
    
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
    print(image.shape)
    plt.figure(figsize=(15,7))
    plt.imshow(image[0:1000,0:1000],cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    imageplot3d(image)
    localstdmean(image,2)
    image= cv2.cvtColor(np.uint8(image),cv2.COLOR_GRAY2RGB)
    hsl=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    Lchannel = hsl[:,:,1]
    mask = cv2.inRange(Lchannel, 250, 255)
    res = cv2.bitwise_and(image,image, mask= mask)
    mask = cv2.inRange(hsl, np.array([0,250,0]), np.array([255,255,255]))
    h,s,l=cv2.split(hsl)
    print()
    print(f'Hue={h.sum()}')
    print(f'Saturation={s.sum()}')
    print(f'Lightness ={l.sum()}')
    print(f'Lightness min ={l.min()}')
    print(f'Lightness max ={l.max()}')
    print(f'Lightness mean ={l.mean()}')
    print(f'Lightness std ={l.std()}')

    
    #print(l.shape)
    #print(type(l))
   
    plt.figure(figsize=(15,7))
    plt.imshow(l[0:1000,0:1000],cmap='gray',vmax=l.max(),vmin=l.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    hsv=cv2.cvtColor(np.uint8(image),cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    print(f'Hue={h.sum()}')
    print(f'Saturation={s.sum()}')
    print(f'Value  ={v.sum()}')
    print(f'Value min ={v.min()}')
    print(f'Value max ={v.max()}')
    print(f'Value mean ={v.mean()}')
    print(f'Value std ={v.std()}')
    print(f'image min ={image.min()}')
    print(f'image max ={image.max()}')
    print(f'image mean ={image.mean()}')
    print(f'image std ={image.std()}')
    #print(v.shape)
    #print(type(v))
    plt.figure(figsize=(15,7))
    plt.imshow(v[0:1000,0:1000],cmap='gray',vmax=v.max(),vmin=v.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    
    image=filtration(image,"C:/Users/evgen/Downloads/s_1_1101_a.jpg")
    #image=filtration(image,"C:/s_1_1101_a.jpg")
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    boxplot_2d(image[0:int(image.shape[0]),:],image[:,0:int(image.shape[1])],ax=ax2, whis=1.5)
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    

    imageground(image)
    
    plt.figure(figsize=(15,7))
    plt.imshow(image[0:1000,0:1000],cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    edges=countourfind(image)
    findcounturnewmethod(edges)
    
    
    
    
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




