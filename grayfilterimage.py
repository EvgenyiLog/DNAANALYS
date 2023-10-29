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

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

from skimage import measure,restoration
from skimage.filters.rank import enhance_contrast
from skimage.morphology import disk
from skimage.filters.rank import autolevel_percentile, autolevel
from skimage.filters.rank import enhance_contrast_percentile
from scipy import ndimage
from skimage.morphology import square,black_tophat,white_tophat

def main():
    image=modelimage()
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    # Adjust the brightness and contrast 
    # Adjusts the brightness by adding 10 to each pixel value 
    brightness = 10 
    # Adjusts the contrast by scaling the pixel values by 2.3 
    contrast = 2.3  
    image1 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
    plt.figure(figsize=(15, 7))
    plt.imshow(image1, cmap=plt.cm.gray,vmax=image1.max(),vmin=image1.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    hist = cv2.calcHist([image],[0],None,[4096],[0,4096])
    print(np.amax(hist))
    print(np.argmax(hist))
    h=list(hist)
    index05max=[]
    for i in h:
        if i>=0.5*np.amax(hist):
            index05max.append(h.index(i))
    index05max=np.asarray(index05max,dtype=np.int64)  
    index99prec=[]
    for i in h:
        if i>=np.percentile(hist,99):
            index99prec.append(h.index(i))
    index05max=np.asarray(index05max,dtype=np.int64)
    index99prec=np.asarray(index99prec,dtype=np.int64)
    #print(index05max.ravel())
    print(np.amax(index05max))
    print(np.amin(index05max))
    print(np.amax(index99prec))
    image2=np.where(image>np.amin(index05max),image,0)
    plt.figure(figsize=(15, 7))
    plt.imshow(image2, cmap=plt.cm.gray,vmax=image2.max(),vmin=image2.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    
    image=cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image3=adjust_gamma(image)
    plt.figure(figsize=(15, 7))
    plt.imshow(image3, cmap=plt.cm.gray,vmax=image3.max(),vmin=image3.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    image=cv2.normalize(image, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image4 = cv2.filter2D(image, -1, sharpen_kernel)
    plt.figure(figsize=(15,7))
    plt.imshow(image4,cmap=plt.cm.gray,vmax=image4.max(),vmin=image4.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    psf = np.ones((1, 1)) / 1
    deconvolved_img = restoration.wiener(image ,psf, 0.0004)
    #deconvolved_img = restoration.unsupervised_wiener(image, psf)
    plt.figure(figsize=(15,7))
    plt.imshow(deconvolved_img,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    # Adjust the brightness and contrast 
    # Adjusts the brightness by adding 10 to each pixel value 
    brightness = 10 
    # Adjusts the contrast by scaling the pixel values by 2.3 
    contrast = 2.3  
    image5 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
    plt.figure(figsize=(15,7))
    plt.imshow(image5,cmap=plt.cm.gray,vmax=image5.max(),vmin=image5.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    # Adjust the brightness and contrast  
    # g(i,j)=α⋅f(i,j)+β 
    # control Contrast by 1.5 
    alpha = 1.5  
    # control brightness by 50 
    beta = 50  
    image6 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    plt.figure(figsize=(15,7))
    plt.imshow(image6,cmap=plt.cm.gray,vmax=image6.max(),vmin=image6.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    # Create the sharpening kernel 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
  
    # Sharpen the image 
    image7 = cv2.filter2D(image, -1, kernel) 
    plt.figure(figsize=(15,7))
    plt.imshow(image7,cmap=plt.cm.gray,vmax=image7.max(),vmin=image7.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    # Apply kernel for embossing 
    emboss_kernel = np.array([[-1, 0, 0], 
                    [0, 0, 0], 
                    [0, 0, 1]]) 
  
    # Embossed image is obtained using the variable emboss_img 
 
    emboss_img = cv2.filter2D(src=image, ddepth=-1, kernel=emboss_kernel) 
    plt.figure(figsize=(15,7))
    plt.imshow(emboss_img,cmap=plt.cm.gray,vmax=emboss_img.max(),vmin=emboss_img.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    image=cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image8 = cv2.equalizeHist(image) 
    image8=cv2.normalize(image8, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    plt.figure(figsize=(15,7))
    plt.imshow(image8,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    
    erosion = cv2.erode(image,kernel,iterations = 1)
    plt.figure(figsize=(15,7))
    plt.imshow(erosion,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    dilation = cv2.dilate(image,kernel,iterations = 1)
    plt.figure(figsize=(15,7))
    plt.imshow(dilation,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    plt.figure(figsize=(15,7))
    plt.imshow(opening,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    plt.figure(figsize=(15,7))
    plt.imshow(closing,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    plt.figure(figsize=(15,7))
    plt.imshow(gradient,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    plt.figure(figsize=(15,7))
    plt.imshow(tophat,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    plt.imshow(blackhat,cmap=plt.cm.gray,vmax=image8.max(),vmin=image8.min())
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений


    enh = enhance_contrast(image, disk(3))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 7),
                         sharex='row', sharey='row')
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    ax[1].imshow(enh, cmap=plt.cm.gray)
    ax[1].set_title('Local morphological contrast enhancement')

    ax[2].imshow(image, cmap=plt.cm.gray)

    ax[3].imshow(enh, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    plt.tight_layout()


    footprint = disk(3)
    loc_autolevel = autolevel(image, footprint=footprint)
    loc_perc_autolevel0 = autolevel_percentile(
    image, footprint=footprint, p0=.01, p1=.99
    )
    loc_perc_autolevel1 = autolevel_percentile(
    image, footprint=footprint, p0=.05, p1=.95
    )
    loc_perc_autolevel2 = autolevel_percentile(
    image, footprint=footprint, p0=.1, p1=.9
    )
    loc_perc_autolevel3 = autolevel_percentile(
    image, footprint=footprint, p0=.15, p1=.85
    )

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10),
                         sharex=True, sharey=True)
    ax = axes.ravel()

    title_list = ['Original',
              'auto_level',
              'auto-level 1%',
              'auto-level 5%',
              'auto-level 10%',
              'auto-level 15%']
    image_list = [image,
              loc_autolevel,
              loc_perc_autolevel0,
              loc_perc_autolevel1,
              loc_perc_autolevel2,
              loc_perc_autolevel3]

    for i in range(0, len(image_list)):
        ax[i].imshow(image_list[i], cmap=plt.cm.gray, vmin=0, vmax=4096)
        ax[i].set_title(title_list[i])
        ax[i].axis('off')

    plt.tight_layout()


    penh = enhance_contrast_percentile(image, disk(3), p0=.1, p1=.9)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 7),
                         sharex='row', sharey='row')
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    ax[1].imshow(penh, cmap=plt.cm.gray)
    ax[1].set_title('Local percentile morphological\n contrast enhancement')

    ax[2].imshow(image, cmap=plt.cm.gray)

    ax[3].imshow(penh, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    plt.tight_layout()

    plt.figure(figsize=(15,7))
    result = ndimage.uniform_filter(image, size=3)
    plt.imshow(result,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений


    result = ndimage.fourier_uniform(image, size=2)
    result = np.fft.ifft2(result)
    plt.figure(figsize=(15,7))
    plt.imshow(result.real,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений




    result=black_tophat(image, square(3))
    plt.figure(figsize=(15,7))
    plt.imshow(result,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений


    result=white_tophat(image, square(3))
    plt.figure(figsize=(15,7))
    plt.imshow(result,cmap='gray')
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

  
    
  
    plt.show()
    
    
if __name__ == "__main__":
    main()

# %%



