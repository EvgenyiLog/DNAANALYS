#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


def modelimage():
    N=int(input('Введите высоту изображения= '))
    M=int(input('Введите ширину изображения= '))
    K1=int(input('Введите количество точек 1= '))
    K2=int(input('Введите количество точек 2= '))
    K3=int(input('Введите количество точек 3= '))
    K4=int(input('Введите количество точек 4= '))
    K5=int(input('Введите количество точек 5= '))
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
    x1=(M+1) * np.random.sample(size=K1)
    y1=(N+1)* np.random.sample(size=K1)
    x1=np.ceil(x1)
    x1=np.asarray(x1,dtype=int)
    y1=np.floor(y1)
    y1=np.asarray(y1,dtype=int)
    k1=y1.shape[0]
    
    image=np.zeros((M,N))
    print(image.shape)
    xx, yy = np.ogrid[0:M,0:N]
    for i in range(k1):
        for  j in range(k1):
            clusters1=(amplitude1/(2*np.pi*np.square(std1)))*np.exp(-(np.square(np.subtract(xx,x1[i]))+np.square(np.subtract(yy,y1[j])))/(2*np.square(std1)))
            
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
            
            image=np.add(image,clusters5)
    image=cv2.normalize(image, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U) 
    
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
    
def main():
    imagem=modelimage()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()
            
            
    


# In[ ]:





# In[ ]:





# In[ ]:




