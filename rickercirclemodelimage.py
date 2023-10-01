#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

def localstdmean(image,N):
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    print(im.shape)
    
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
    cv2.imwrite("C:/Users/evgen/Downloads/noise_model.jpg",noises)
    background=scipy.signal.convolve2d(im, kernel, mode="same")
    background=np.asarray(background,dtype=np.uint8)
    backgrounds=cv2.cvtColor( background, cv2.COLOR_GRAY2BGR) 
    cv2.imwrite("C:/Users/evgen/Downloads/background_model.jpg",backgrounds)
    plt.figure(figsize=(15, 7))
    plt.imshow(background[0:1000,0:1000], cmap=plt.cm.gray,vmax=background.max(),vmin=background.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    print(f'image min ={image.min()}')
    print(f'image max ={image.max()}')
    print(f'image mean ={image.mean()}')
    print(f'image std ={image.std()}')
    
    
    

def modelimage():
    N=int(input('Введите высоту изображения= '))
    M=int(input('Введите ширину изображения= '))
    K1=int(input('Введите количество точек 1= '))
    K2=int(input('Введите количество точек 2= '))
    K3=int(input('Введите количество точек 3= '))
    R1=int(input('Введите радиус 1='))
    R2=int(input('Введите радиус 2='))
    R3=int(input('Введите радиус 3='))
    amplitude1=int(input('Введите амплитуду 1='))
    amplitude2=int(input('Введите амплитуду 2='))
    amplitude3=int(input('Введите амплитуду 3='))
    std1=int(input('Введите среднеквадратичное отклонение 1='))
    std2=int(input('Введите среднеквадратичное отклонение 2='))
    std3=int(input('Введите среднеквадратичное отклонение 3='))
    image=np.zeros((M,N))
    print(image.shape)
    xx, yy = np.ogrid[0:M,0:N]
    xcentr1=(M+1) * np.random.sample(size=K1)
    ycentr1=(N+1)* np.random.sample(size=K1)
    xcentr1=np.ceil(xcentr1)
    xcentr1=np.asarray(xcentr1,dtype=int)
    ycentr1=np.floor(ycentr1)
    ycentr1=np.asarray(ycentr1,dtype=int)
    #print(ycentr1)
    #print(xcentr1)
    xcentr2=(M+1) * np.random.sample(size=K2)
    ycentr2=(N+1)* np.random.sample(size=K2)
    xcentr2=np.ceil(xcentr2)
    xcentr2=np.asarray(xcentr2,dtype=int)
    ycentr2=np.floor(ycentr2)
    ycentr2=np.asarray(ycentr2,dtype=int)
    xcentr3=(M+1) * np.random.sample(size=K3)
    ycentr3=(N+1)* np.random.sample(size=K3)
    xcentr3=np.ceil(xcentr3)
    xcentr3=np.asarray(xcentr3,dtype=int)
    ycentr3=np.floor(ycentr3)
    ycentr3=np.asarray(ycentr3,dtype=int)
                       
    
    image=np.asarray(image,dtype=np.uint8)
    
    for i in range(K1):
        imagem=cv2.circle(image, (xcentr1[i], ycentr1[i]), R1, (255, 255, 255), -1)
    for i in range(K2):
        imagem=cv2.circle(imagem, (xcentr2[i], ycentr2[i]), R2, (200, 200, 200), -1)
    for i in range(K3):
        imagem=cv2.circle(imagem, (xcentr3[i], ycentr3[i]), R3, (240, 240, 240), -1)
        
    x1=(M+1) * np.random.sample(size=K1)
    y1=(N+1)* np.random.sample(size=K1)
    x1=np.ceil(x1)
    x1=np.asarray(x1,dtype=int)
    y1=np.floor(y1)
    y1=np.asarray(y1,dtype=int)
    k1=y1.shape[0]
    image=np.zeros((N,M))
    for i in range(k1):
        for  j in range(k1):
            clusters1=amplitude1*(1-(np.square(np.subtract(xx,x1[i]))+np.square(np.subtract(yy,y1[j])))/np.square(std1))*np.exp(-(np.square(
            np.subtract(xx,x1[i]))+np.square(np.subtract(yy,y1[j])))/2*np.square(std1))
            image=np.add(image,clusters1)
    #x2=np.arange(centr2,M,centr2)
                                                                                                
    #y2=np.arange(centr2,N,centr2)
    #print(f'Количество точек 2={len(x2)}')
    x2=(M+1) * np.random.sample(size=K2)
    y2=(N+1)* np.random.sample(size=K2)
    x2=np.ceil(x2)
    x2=np.asarray(x2,dtype=int)
    y2=np.floor(y2)
    y2=np.asarray(y2,dtype=int)
    k2=y2.shape[0]
    for i in range(k2):
        for j in range(k2):
            clusters2=amplitude2*(1-(np.square(np.subtract(xx,x2[i]))+np.square(np.subtract(yy,y2[j])))/np.square(std2))*np.exp(-(np.square(
            np.subtract(xx,x2[i]))+np.square(np.subtract(yy,y2[j])))/2*np.square(std2))
            image=np.add(image,clusters2)
        
    #x3=np.arange(centr3,M,centr3)
                                                                                                                                                                             
    
    #y3=np.arange(centr3,N,centr3)
    #print(f'Количество точек 3={len(x3)}')
    x3=(M+1) * np.random.sample(size=K3)
    y3=(N+1)* np.random.sample(size=K3)
    x3=np.ceil(x3)
    x3=np.asarray(x3,dtype=int)
    y3=np.floor(y3)
    y3=np.asarray(y3,dtype=int)
    k3=y3.shape[0]
    for i in range(k3):
        for j in range(k3):
            clusters3=amplitude3*(1-(np.square(np.subtract(xx,x3[i]))+np.square(np.subtract(yy,y3[j])))/np.square(std3))*np.exp(-(np.square(
            np.subtract(xx,x3[i]))+np.square(np.subtract(yy,y3[j])))/2*np.square(std3))
            image=np.add(image,clusters3)
    imagem=cv2.normalize(imagem, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    image=cv2.normalize(image, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    image=cv2.add(image,imagem)
    image= cv2.blur(image,(3,3))
    image = cv2.GaussianBlur(image, (3,3),0)
    kernel = np.ones((3,3),np.float32)/9
    image = cv2.filter2D(image,-1,kernel)
    image = cv2.medianBlur(image,3)
    image = cv2.boxFilter(image, -1, (3,3))
    print(image.dtype)
    print(image.min())
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d                                                                                                              fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx,yy,image)
    ax.grid(True)
           
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    
    fig=plt.figure(figsize=(15,7))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xx,yy,image)
    ax.grid(True)
    ay = fig.add_subplot(122)
    ay.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    #ay.grid(True)
    ay.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет делений  
    plt.savefig("C:/Users/evgen/Downloads/rickercirclenoisemodel.jpg")
                                                                                                                     
                                                                                                                     
def main():
    imagem=modelimage()
    
    plt.show()
    
    
if __name__ == "__main__":
    main() 


# In[ ]:





# In[ ]:





# In[ ]:




