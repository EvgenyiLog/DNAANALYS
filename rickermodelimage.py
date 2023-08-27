#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
    
    
    
def rickermodelimage():
    N=int(input('Введите высоту изображения='))
    M=int(input('Введите ширину изображения='))
    amplitude1=int(input('Введите амплитуду 1='))
    amplitude2=int(input('Введите амплитуду 2='))
    amplitude3=int(input('Введите амплитуду 3='))
    #centr1=int(input('Введите значение центра 1='))
    #centr2=int(input('Введите значение центра 2='))
    #centr3=int(input('Введите значение центра 3='))
    K1=int(input('Введите количество точек 1='))
    K2=int(input('Введите количество точек 2='))
    K3=int(input('Введите количество точек 3='))
    std1=int(input('Введите среднеквадратичное отклонение 1='))
    std2=int(input('Введите среднеквадратичное отклонение 2='))
    std3=int(input('Введите среднеквадратичное отклонение 3='))
    xx, yy = np.ogrid[0:M,0:N]
    #x1=np.arange(centr1,M,centr1)
    
   
    #y1=np.arange(centr1,N,centr1)
    #print(f'Количество точек 1={len(x1)}')
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
    
    
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d                                                                                                              fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx,yy,image)
    ax.grid(True)
    image=np.asarray(image,dtype=np.uint16)         
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray',vmax=image.max(),vmin=image.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений   
                                                                                                                     
                                                                                                                     
def main():
    imagem=rickermodelimage()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()                                                                                                                    
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
                                                                                                    
                                                                                                    
                                                                                                    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




