#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
def modelimage():
    N=int(input('Введите высоту изображения'))
    M=int(input('Введите ширину изображения'))
    K1=int(input('Введите количество точек 1'))
    K2=int(input('Введите количество точек 2'))
    K3=int(input('Введите количество точек 3'))
    K4=int(input('Введите количество точек 4'))
    K5=int(input('Введите количество точек 5'))
    R1=int(input('Введите радиус 1'))
    R2=int(input('Введите радиус 2'))
    R3=int(input('Введите радиус 3'))
    R4=int(input('Введите радиус 4'))
    R5=int(input('Введите радиус 5'))
    image=np.zeros((M,N,3))
    print(image.shape)
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
    xcentr4=(M+1) * np.random.sample(size=K4)
    ycentr4=(N+1)* np.random.sample(size=K4)
    xcentr4=np.ceil(xcentr4)
    xcentr4=np.asarray(xcentr4,dtype=int)
    ycentr4=np.floor(ycentr4)
    ycentr4=np.asarray(ycentr4,dtype=int)
    xcentr5=(M+1) * np.random.sample(size=K5)
    ycentr5=(N+1)* np.random.sample(size=K5)
    xcentr5=np.ceil(xcentr5)
    xcentr5=np.asarray(xcentr5,dtype=int)
    ycentr5=np.floor(ycentr5)
    ycentr5=np.asarray(ycentr5,dtype=int)
                      
                       
    
    image=np.asarray(image,dtype=np.uint8)
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    for i in range(K1):
        imagem=cv2.circle(image, (xcentr1[i], ycentr1[i]), R1, (255, 255, 255), -1)
    for i in range(K2):
        imagem=cv2.circle(imagem, (xcentr2[i], ycentr2[i]), R2, (200, 200, 200), -1)
    for i in range(K3):
        imagem=cv2.circle(imagem, (xcentr3[i], ycentr3[i]), R3, (240, 240, 240), -1)
    for i in range(K4):
        imagem=cv2.circle(imagem, (xcentr4[i], ycentr4[i]), R4, (240, 240, 240), -1)
    for i in range(K5):
        imagem=cv2.circle(imagem, (xcentr5[i], ycentr5[i]), R5, (240, 240, 240), -1)
    
                          
    plt.figure(figsize=(15, 7))
    plt.imshow(imagem, cmap=plt.cm.gray,vmax=imagem.max(),vmin=imagem.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    imagem=cv2.cvtColor(imagem,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("C:/Users/evgen/Downloads/imagem.jpg",imagem)
    cv2.imwrite("C:/Users/evgen/Downloads/imagem.tiff",imagem)
    return imagem

import statistics   
def groundimage(image):
    # Вычисляем маску фона
    image=np.uint8(image)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Убираем шум
    kernel = np.ones((2, 2), np.uint16)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=50)
    background=opening
    
    background=cv2.normalize(background, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U) 
    # создаем маску фона
    sure_bg = cv2.dilate(opening, kernel, iterations=20) 


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  

    foreground=sure_fg
    
    foreground=cv2.normalize(foreground, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    unknown = cv2.subtract(background,foreground)
    unknown=cv2.normalize(unknown, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    #print(background.dtype)
    #print(foreground.dtype)
    #print(type(background))
    #print(type(foreground))
    return np.sum(foreground)

def findcountour(image):
    imagesource=image
    'поиск контуров'
    edges = cv2.Canny(image=image, threshold1=1, threshold2=255)
   
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
    widthc=[] 
    heightc=[]
    angle=[]
    
    ret, thresh = cv2.threshold(edges, 1, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    
    for num,i in enumerate(contours):
        M = cv2.moments(i)
        (x,y), (width, height), angle= cv2.minAreaRect(i)
       
        #print(x)
        #print(y)
        #print(width)
        #print(height)
        xcentrrect.append(x)
        ycentrrect.append(y)
        widthc.append(width)
        heightc.append(height) 
        intensivityrect.append(imagesource[int(np.floor(y)),int(np.floor(x))])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr.append(cx)
            ycentr.append(cy)
            intensivity.append(imagesource[cy,cx])
    print('Количество')
    print(len(contours))
    print(len(xcentr))
    print()
    return xcentr,ycentr,xcentrrect,ycentrrect, widthc, heightc
from skimage.feature import peak_local_max
def main():
    image0=modelimage()
    coordinates = peak_local_max(image0)
    im=image0[coordinates[:, 0], coordinates[:, 1]]
    im=np.sort(im)
    image=im[0]*image0/(im[0]+im[1])
    print(image.max())
    image=np.rint(image)
    image=np.asarray(image,dtype=np.uint8)
    #image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                   color = 'k')   #  Цвет делений
    '''
    xcentr,ycentr,xcentrrect,ycentrrect, width, height=findcountour(image)
    i1= groundimage(image[xcentr[2]: xcentr[2]+height[2],ycentr[2]: ycentr[2]+width[2]])
    i2= groundimage(image[xcentr[3]: xcentr[3]+height[3],ycentr[3]: ycentr[3]+width[3]])
    image=i1*image0/(i1+i2)
    plt.figure(figsize=(15,7))
    plt.imshow(image,cmap='gray')
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                   color = 'k')   #  Цвет делений
    '''
    
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()


# In[ ]:




