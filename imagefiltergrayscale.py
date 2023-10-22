#!/usr/bin/env python
# coding: utf-8

# In[16]:


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

from skimage.filters import threshold_multiotsu
from skimage.filters import meijering,sato,gabor,butterworth
from skimage.filters.thresholding import _cross_entropy
from skimage import filters
def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-10-09_21-50-34.jpg")
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))

    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

   # Plotting the histogram and the two thresholds obtained from
   # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    ax[1].grid(True)
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()
    
    imagem=meijering(image)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagem, cmap=plt.cm.gray,vmax=imagem.max(),vmin=imagem.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    images=sato(image)
    plt.figure(figsize=(15, 7))
    plt.imshow(images, cmap=plt.cm.gray,vmax=images.max(),vmin=images.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    thresholds = np.arange(np.min(image) + 1.5, np.max(image) - 1.5)
    entropies = [_cross_entropy(image, t) for t in thresholds]

    optimal_image_threshold = thresholds[np.argmin(entropies)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('image')
    ax[0].set_axis_off()

    ax[1].imshow(image > optimal_image_threshold, cmap='gray')
    ax[1].set_title('thresholded')
    ax[1].set_axis_off()

    ax[2].plot(thresholds, entropies)
    ax[2].grid(True)
    ax[2].set_xlabel('thresholds')
    ax[2].set_ylabel('cross-entropy')
    ax[2].vlines(optimal_image_threshold,
             ymin=np.min(entropies) - 0.05 * np.ptp(entropies),
             ymax=np.max(entropies) - 0.05 * np.ptp(entropies))
    ax[2].set_title('optimal threshold')

    fig.tight_layout()

    print('The brute force optimal threshold is:', optimal_image_threshold)
    print('The computed optimal threshold is:', filters.threshold_li(image))
    
    filt_real, filt_imag = gabor(image, frequency=0.6)
    image1=filt_real
    image2=filt_imag
    plt.figure(figsize=(15, 7))
    plt.imshow(image1, cmap=plt.cm.gray,vmax=image1.max(),vmin=image1.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.figure(figsize=(15, 7))
    plt.imshow(image2, cmap=plt.cm.gray,vmax=image2.max(),vmin=image2.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    high_pass=butterworth(image, 0.07, True, 8)
    image3=high_pass
    plt.figure(figsize=(15, 7))
    plt.imshow(image3, cmap=plt.cm.gray,vmax=image3.max(),vmin=image3.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    


    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




