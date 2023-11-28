#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
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


def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1


            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image

def main():
    image=modelimage()
    g = np.gradient(image)
    g=np.add(g[0],g[1])
    plt.figure(figsize=(15,7))
    plt.imshow(g, aspect='auto',cmap='gray')
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    g01=Zero_crossing(g)
    plt.figure(figsize=(15,7))
    plt.imshow(g01, aspect='auto',cmap='gray')
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    cv2.imwrite("C:/Users/evgen/Downloads/g01.jpg",g01)
    cv2.imwrite("C:/Users/evgen/Downloads/g01.tiff",g01)
    g = np.gradient(g)
    g=np.add(g[0],g[1])
    plt.figure(figsize=(15,7))
    plt.imshow(g, aspect='auto',cmap='gray')
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    g02=Zero_crossing(g)
    plt.figure(figsize=(15,7))
    plt.imshow(g02, aspect='auto',cmap='gray')
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    cv2.imwrite("C:/Users/evgen/Downloads/g02.jpg",g02)
    cv2.imwrite("C:/Users/evgen/Downloads/g02.tiff",g02)
    
    
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




