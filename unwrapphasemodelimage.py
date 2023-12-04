#!/usr/bin/env python
# coding: utf-8

# In[2]:


from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
import numpy as np
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

def main():
    image=modelimage()
    # Scale the image to [0, 4*pi]
    image = exposure.rescale_intensity(image, out_range=(0, 4 * np.pi))
    # Create a phase-wrapped image in the interval [-pi, pi)
    image_wrapped = np.angle(np.exp(1j * image))
    # Perform phase unwrapping
    image_unwrapped = unwrap_phase(image_wrapped)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1, ax2, ax3, ax4 = ax.ravel()

    fig.colorbar(ax1.imshow(image, cmap='gray', vmin=0, vmax=4 * np.pi), ax=ax1)
    ax1.set_title('Original')
    ax1.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    fig.colorbar(ax2.imshow(image_wrapped, cmap='gray', vmin=-np.pi, vmax=np.pi),
             ax=ax2)
    ax2.set_title('Wrapped phase')
    ax2.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    fig.colorbar(ax3.imshow(image_unwrapped, cmap='gray'), ax=ax3)
    ax3.set_title('After phase unwrapping')
    ax3.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    fig.colorbar(ax4.imshow(image_unwrapped - image, cmap='gray'), ax=ax4)
    ax4.set_title('Unwrapped minus original')
    ax4.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()

    


# In[ ]:




