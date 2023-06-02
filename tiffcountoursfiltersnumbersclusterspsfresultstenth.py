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
    image=np.asarray(image,dtype=np.uint8)
    return image


def filtration(image,path):
    'path to file correlate'
    'image filtration'
    'фильтрация возвращает фильтрованное изображение'
    psnr=peak_signal_noise_ratio(image, std_convoluted(image,2))
    print(f'PSNR before filtration={psnr}')
    
    image=scipy.signal.wiener(image,noise=image.std())
    
    
    image=np.asarray(image,dtype=np.uint8)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image= cv2.bilateralFilter(image,1,1,1)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_wiener.jpg",image)
    slp,hlp=sporco.signal.tikhonov_filter(image,2)
    hlp=ndimage.median_filter(hlp, size=7)
    hlp=hlp<np.percentile(hlp, 95)
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
    eng = matlab.engine.start_matlab()   
    r=eng.corr2(hlp,image2)
    hlp=hlp-hlp.mean()-r*image2
    
    
    hlpfilt=np.asarray(hlp,dtype=np.uint8)
    #hlpfilt = eng.imadjust(hlpfilt,eng.stretchlim(hlpfilt),[])
    hlpfilt=eng.imadjust(hlpfilt)
    hlpfilt=np.asarray(hlpfilt,dtype=np.uint8)
    hlpfilt=cv2.cvtColor(hlpfilt,cv2.COLOR_GRAY2RGB)
    #hlp=scipy.ndimage.percentile_filter(hlp,95)
    
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_cfilt.jpg",hlpfilt)
    #cv2.imwrite("C:/Users/Евгений/Downloads/s_1_1102_cfilt.jpg",hlpfilt)
    r=eng.normxcorr2(hlp,image2)
    r=np.asarray(r,dtype=np.uint8)
    print(r.shape)
    r=cv2.cvtColor(r, cv2.COLOR_GRAY2BGR) 
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
    
    sigma_est =skimage.restoration.estimate_sigma(hlpfilt)
    imagew=skimage.restoration.denoise_wavelet(hlpfilt,sigma=sigma_est)
    plt.figure(figsize=(15,7))
    plt.imshow(imagew[0:1000,0:1000],cmap='gray',vmax=imagew.max(),vmin=imagew.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    psnr=peak_signal_noise_ratio(hlpfilt, std_convoluted(hlpfilt,2))
    print(f'PSNR after filtration={psnr}')
    return hlpfilt

from scipy import special
import pylab
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy import ndimage as ndi
from skimage import filters
from scipy import ndimage

def countourfind(image):
    imagesource=image
    'поиск контуров'
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

    

    edges = cv2.Canny(image=image, threshold1=1, threshold2=4)
    plt.figure(figsize=(15,7))
    plt.imshow(edges[0:1000,0:1000],cmap='gray',vmax=edges.max(),vmin=edges.min())
    plt.plot(coordinates[0:1000, 1], coordinates[0:1000, 0], 'r.')
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
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
    autoencoder.fit(x_train, x_train,epochs=200,batch_size=512)
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
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print('Координаты центра')
    xcentr=[]
    ycentr=[]
    
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr.append(cx)
            ycentr.append(cy)
            
            
        
    print('Количество')
    print(len(xcentr))
    print()
    
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
    
    image5=cv2.circle(image3, (cx, cy), 1, (255, 255, 255), -1)
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
    ln=np.zeros_like(l)
    ln[ycentr,xcentr]=l[ycentr,xcentr]
    ln=np.asarray(ln,dtype=np.uint8)
    image7=cv2.merge([h,s,ln])
    image7=cv2.cvtColor(image7,cv2.COLOR_HLS2BGR)
    plt.figure(figsize=(15,7))
    plt.imshow(image7[0:1000,0:1000],cmap='gray',vmax=image7.max(),vmin=image7.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    vn=np.zeros_like(v)
    vn[ycentr,xcentr]=v[ycentr,xcentr]
    vn=np.asarray(vn,dtype=np.uint8)
    image8=cv2.merge([h,s,vn])
    image8=cv2.cvtColor(image8,cv2.COLOR_HLS2BGR)
    plt.figure(figsize=(15,7))
    plt.imshow(image8[0:1000,0:1000],cmap='gray',vmax=image8.max(),vmin=image8.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
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
    
    
    
   
    
    
    
    
    
def main():
    image=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    #image=readimage("C:/s_1_1102_c.jpg")
    #image=readimage("C:/s_1_1101_a.jpg")
    #image=image[0:1000,0:1000]
    print(image.shape)
    plt.figure(figsize=(15,7))
    plt.imshow(image[0:1000,0:1000],cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    image= cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
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
    #print(l.shape)
    #print(type(l))
   
    plt.figure(figsize=(15,7))
    plt.imshow(l[0:1000,0:1000],cmap='gray',vmax=l.max(),vmin=l.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    print(f'Hue={h.sum()}')
    print(f'Saturation={s.sum()}')
    print(f'Value  ={v.sum()}')
    print(f'Value min ={v.min()}')
    print(f'Value max ={v.max()}')
    #print(v.shape)
    #print(type(v))
    plt.figure(figsize=(15,7))
    plt.imshow(v[0:1000,0:1000],cmap='gray',vmax=v.max(),vmin=v.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    
    image=filtration(image,"C:/Users/evgen/Downloads/s_1_1101_a.jpg")
    #image=filtration(image,"C:/s_1_1101_a.jpg")
    
    plt.figure(figsize=(15,7))
    plt.imshow(image[0:1000,0:1000],cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    countourfind(image)
    
    
    plt.show()
    
    
if __name__ == "__main__":
    main()