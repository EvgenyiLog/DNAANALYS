# %%
import numpy
from numpy.fft import fft, ifft
from matplotlib import pyplot as plt
import numpy as np
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def readimage(path):
    'path to file'
    'чтение файла возвращает изображение'
    
    image = cv2.imread(path,0)
    #print(image.shape)
    h, w = image.shape[:2]
    print(f'Weigth= {w}')
    print(f'Heigth= {h}')
    print(image.dtype)
    return image

def tiffreader(path):
    image=cv2.imread(path,-1)
    #print(image.shape)
    print(image.dtype)
    h, w = image.shape[:2]
    print(f'Weigth= {w}')
    print(f'Heigth= {h}')
    return image

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
        
        
    for i in range(K1):
        imagem=cv2.circle(image, (xcentr1[i], ycentr1[i]), 1, (255, 0, 0), -1)
    for i in range(K2):
        imagem=cv2.circle(imagem, (xcentr2[i], ycentr2[i]), 1, (255, 0, 0), -1)
    for i in range(K3):
        imagem=cv2.circle(imagem, (xcentr3[i], ycentr3[i]), 1, (255, 0, 0), -1)
        
    for i in range(K4):
        imagem=cv2.circle(imagem, (xcentr4[i], ycentr4[i]), 1, (255, 0, 0), -1)
        
    for i in range(K5):
        imagem=cv2.circle(imagem, (xcentr5[i], ycentr5[i]), 1, (255, 0, 0), -1)
        
    imagem=cv2.normalize(imagem, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        
    imagem= cv2.blur(imagem,(3,3))
    imagem = cv2.GaussianBlur(imagem, (3,3),0)
    kernel = np.ones((3,3),np.float32)/9
    imagem = cv2.filter2D(imagem,-1,kernel)
    imagem = cv2.medianBlur(imagem,3)
    imagem = cv2.boxFilter(imagem, -1, (3,3))
    
    print(imagem.dtype)
        
        
    plt.figure(figsize=(15, 7))
    plt.imshow(imagem, cmap=plt.cm.gray,vmax=imagem.max(),vmin=imagem.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    
    
    
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d                                                                                                              fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx,yy,imagem)
    ax.grid(True)
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
from skimage import filters
def laplacian_of_gaussian(image, sigma):
    """
    Applies a Gaussian kernel to an image and the Laplacian afterwards.
    """
    
    # blur the image using a Gaussian kernel
    intermediate_result = filters.gaussian(image, sigma)
    
    # apply the mexican hat filter (Laplacian)
    result = filters.laplace(intermediate_result)
    
    return result
from scipy import ndimage
from skimage.filters import difference_of_gaussians

from skimage.filters import threshold_otsu, threshold_niblack,threshold_sauvola
def main():
    image=modelimage()
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    l=cv2.Laplacian(image, cv2.CV_64F)
    print(l.max())
    print(l.min())
    plt.figure(figsize=(15, 7))
    plt.imshow(l, cmap=plt.cm.gray,vmax=l.max(),vmin=l.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    lzc=Zero_crossing(l)
    print(lzc.max())
    print(lzc.min())
    plt.figure(figsize=(15, 7))
    plt.imshow(lzc, cmap=plt.cm.gray,vmax=lzc.max(),vmin=lzc.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    l= cv2.convertScaleAbs(l)
    print(l.max())
    print(l.min())
    plt.figure(figsize=(15, 7))
    plt.imshow(l, cmap=plt.cm.gray,vmax=l.max(),vmin=l.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    binary_global = image > threshold_otsu(image)

    window_size = 3
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Global Threshold')
    plt.imshow(binary_global, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(binary_niblack, cmap=plt.cm.gray)
    plt.title('Niblack Threshold')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(binary_sauvola, cmap=plt.cm.gray)
    plt.title('Sauvola Threshold')
    plt.axis('off')
    
    result=ndimage.gaussian_laplace(image, sigma=1)
    print(result.max())
    print(result.min())
    plt.figure(figsize=(15, 7))
    plt.imshow(result, cmap=plt.cm.gray,vmax=result.max(),vmin=result.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    log=laplacian_of_gaussian(image,1)
    print(log.max())
    print(log.min())
    plt.figure(figsize=(15, 7))
    plt.imshow(log, cmap=plt.cm.gray,vmax=log.max(),vmin=log.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    log=difference_of_gaussians(image, 1, 10,channel_axis=-1)
    plt.figure(figsize=(15, 7))
    plt.imshow(log, cmap=plt.cm.gray,vmax=log.max(),vmin=log.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.show()
    
    
if __name__ == "__main__":
    main()

# %%



