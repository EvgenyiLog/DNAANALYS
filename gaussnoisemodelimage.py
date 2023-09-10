#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from PIL import Image
import opensimplex as simplex


def modelimage():
    N=int(input('Введите высоту изображения= '))
    M=int(input('Введите ширину изображения= '))
    K1=int(input('Введите количество точек 1= '))
    K2=int(input('Введите количество точек 2= '))
    K3=int(input('Введите количество точек 3= '))
    K4=int(input('Введите количество точек 4= '))
    K5=int(input('Введите количество точек 5= '))
    R1=int(input('Введите радиус 1='))
    R2=int(input('Введите радиус 2='))
    R3=int(input('Введите радиус 3='))
    R4=int(input('Введите радиус 4='))
    R5=int(input('Введите радиус 5='))
    amplitude1=int(input('Введите амплитуду 1='))
    amplitude2=int(input('Введите амплитуду 2='))
    amplitude3=int(input('Введите амплитуду 3='))
    amplitude4=int(input('Введите амплитуду 4='))
    amplitude5=int(input('Введите амплитуду 5='))
    std1=int(input('Введите среднеквадратичное отклонение 1='))
    std2=int(input('Введите среднеквадратичное отклонение 2='))
    std3=int(input('Введите среднеквадратичное отклонение 3='))
    std4=int(input('Введите среднеквадратичное отклонение 4='))
    std5=int(input('Введите среднеквадратичное отклонение 5='))
    whitenoisevalue=int(input('Введите уровень белого шума='))
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
    
    for i in range(K1):
        imagem=cv2.circle(image, (xcentr1[i], ycentr1[i]), R1, (255, 255, 255), -1)
    for i in range(K2):
        imagem=cv2.circle(imagem, (xcentr2[i], ycentr2[i]), R2, (200, 200, 200), -1)
    for i in range(K3):
        imagem=cv2.circle(imagem, (xcentr3[i], ycentr3[i]), R3, (240, 240, 240), -1)
        
    for i in range(K4):
        imagem=cv2.circle(imagem, (xcentr4[i], ycentr4[i]), R4, (230, 230, 230), -1)
        
    for i in range(K5):
        imagem=cv2.circle(imagem, (xcentr5[i], ycentr5[i]), R5, (220, 220, 220), -1)
        
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
            clusters1=(amplitude1/(2*np.pi*np.square(std1)))*np.exp(-(np.square(np.subtract(xx,x1[i]))+np.square(np.subtract(yy,y1[j])))/(2*np.square(std1)))
            #clusters1=amplitude1*(1-(np.square(np.subtract(xx,x1[i]))+np.square(np.subtract(yy,y1[j])))/np.square(std1))*np.exp(-(np.square(
            #np.subtract(xx,x1[i]))+np.square(np.subtract(yy,y1[j])))/2*np.square(std1))
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
            clusters2=(amplitude2/(2*np.pi*np.square(std2)))*np.exp(-(np.square(np.subtract(xx,x2[i]))+np.square(np.subtract(yy,y2[j])))/(2*np.square(std2)))
            #clusters2=amplitude2*(1-(np.square(np.subtract(xx,x2[i]))+np.square(np.subtract(yy,y2[j])))/np.square(std2))*np.exp(-(np.square(
            #np.subtract(xx,x2[i]))+np.square(np.subtract(yy,y2[j])))/2*np.square(std2))
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
            clusters3=(amplitude3/(2*np.pi*np.square(std3)))*np.exp(-(np.square(np.subtract(xx,x3[i]))+np.square(np.subtract(yy,y3[j])))/(2*np.square(std3)))
            #clusters3=amplitude3*(1-(np.square(np.subtract(xx,x3[i]))+np.square(np.subtract(yy,y3[j])))/np.square(std3))*np.exp(-(np.square(
            #np.subtract(xx,x3[i]))+np.square(np.subtract(yy,y3[j])))/2*np.square(std3))
            image=np.add(image,clusters3)
            
            
    x4=(M+1) * np.random.sample(size=K4)
    y4=(N+1)* np.random.sample(size=K4)
    x4=np.ceil(x4)
    x4=np.asarray(x4,dtype=int)
    y4=np.floor(y4)
    y4=np.asarray(y4,dtype=int)
    k4=y4.shape[0]
    for i in range(k4):
        for j in range(k4):
            clusters4=(amplitude4/(2*np.pi*np.square(std4)))*np.exp(-(np.square(np.subtract(xx,x4[i]))+np.square(np.subtract(yy,y4[j])))/(2*np.square(std4)))
            #clusters4=amplitude4*(1-(np.square(np.subtract(xx,x4[i]))+np.square(np.subtract(yy,y4[j])))/np.square(std4))*
            #np.exp(-(np.square(np.subtract(xx,x4[i]))+np.square(np.subtract(yy,y4[j])))/2*np.square(std4))
            image=np.add(image,clusters4)
            
            
    x5=(M+1) * np.random.sample(size=K5)
    y5=(N+1)* np.random.sample(size=K5)
    x5=np.ceil(x5)
    x5=np.asarray(x5,dtype=int)
    y5=np.floor(y5)
    y5=np.asarray(y5,dtype=int)
    k5=y5.shape[0]
    for i in range(k5):
        for j in range(k5):
            clusters5=(amplitude5/(2*np.pi*np.square(std5)))*np.exp(-(np.square(np.subtract(xx,x5[i]))+np.square(np.subtract(yy,y5[j])))/(2*np.square(std5)))
            #clusters5=amplitude5*(1-(np.square(np.subtract(xx,x5[i]))+np.square(np.subtract(yy,y5[j])))/np.square(std5))*
            #np.exp(-(np.square(np.subtract(xx,x5[i]))+np.square(np.subtract(yy,y5[j])))/2*np.square(std5))
            image=np.add(image,clusters5)
    imagem=cv2.normalize(imagem, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    image=cv2.normalize(image, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    noise1 = PerlinNoise(octaves=3)
    noise2 = PerlinNoise(octaves=6)
    noise3 = PerlinNoise(octaves=12)
    noise4 = PerlinNoise(octaves=24)
    noise5 = PerlinNoise(octaves=48)
    xpix, ypix = M, N
    pic = []
    for i in range(xpix):
        row = []
        for j in range(ypix):
            noise_val = noise1([i/xpix, j/ypix])
            noise_val += 0.0005 * noise2([i/xpix, j/ypix])
            noise_val += 0.00025 * noise3([i/xpix, j/ypix])
            noise_val += 0.000125 * noise4([i/xpix, j/ypix])
            noise_val += 0.0000625 * noise5([i/xpix, j/ypix])

            row.append(noise_val)
        pic.append(row)
        
    WIDTH = M
    HEIGHT = N
    FEATURE_SIZE = 24.0



    im = Image.new('L', (WIDTH, HEIGHT))
    for y in range(0, HEIGHT):
        for x in range(0, WIDTH):
            value = simplex.noise2(x / FEATURE_SIZE, y / FEATURE_SIZE)
            color = int((value + 1) * 128/500)
            im.putpixel((x, y), color)
    whitenoise=np.random.randint(whitenoisevalue, size=(M, N))
    whitenoise=np.asarray(whitenoise,dtype=np.uint16)
    im=np.asarray(im,dtype=np.uint16)
    pic=np.asarray(pic,dtype=np.uint16)

    image=cv2.add(image,imagem)
    image=cv2.add(image,pic)
    image=cv2.add(image,im)
    image=cv2.add(image,whitenoise)
    image= cv2.blur(image,(3,3))
    image = cv2.GaussianBlur(image, (3,3),0)
    kernel = np.ones((3,3),np.float32)/9
    image = cv2.filter2D(image,-1,kernel)
    image = cv2.medianBlur(image,3)
    image = cv2.boxFilter(image, -1, (3,3))




    print(image.dtype)
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
    plt.savefig("C:/Users/evgen/Downloads/gausscirclenoisemodel.jpg")

    
    
                                                   
        
    
                                                                                                                     
def main():
    imagem=modelimage()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




