#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[25]:


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


def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters


def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage


from skimage.transform import probabilistic_hough_line
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.filters import hessian
import skimage
def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-10-09_21-50-34.jpg")
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    # We create our gabor filters, and then apply them to our image
    gfilters = create_gaborfilter()
    image_g = apply_filter(image, gfilters)
    plt.figure(figsize=(15, 7))
    plt.imshow(image_g, cmap=plt.cm.gray,vmax=image_g.max(),vmin=image_g.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    
    imagec=image.copy()
    edges = cv2.Canny(image,0,256,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,0,minLineLength=1,maxLineGap=1000)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(imagec,(x1,y1),(x2,y2),(255,255,255),1)
        
        
    plt.figure(figsize=(15, 7))
    plt.imshow(imagec, cmap=plt.cm.gray,vmax=imagec.max(),vmin=imagec.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    
    lines = probabilistic_hough_line(edges, threshold=1, line_length=5,
                                 line_gap=3)
    plt.figure(figsize=(15, 7))
    plt.imshow(edges,cmap= 'gray')
    for line in lines:
        p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
    
    
    blobs_log = blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_dog = blob_dog(image, max_sigma=30, threshold=.1)
    blobs_doh = blob_doh(image, max_sigma=30, threshold=.01)
    


    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
    sequence = zip(blobs_list,titles)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharex=True, sharey=True)
    ax = axes.ravel()
    for idx, (blobs, title) in enumerate(sequence):                                                                                      
        ax[idx].set_title(title,fontsize=20)
        ax[idx].imshow(image,cmap='gray')
        ax[idx].tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений 
        ax[idx].set_axis_off()

    plt.tight_layout()
    
    imageh=hessian(image,[1])
    plt.figure(figsize=(15, 7))
    plt.imshow(imageh, cmap=plt.cm.gray,vmax=imageh.max(),vmin=imageh.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    imagefv=skimage.filters.farid_v(image) 
    plt.figure(figsize=(15, 7))
    plt.imshow(imagefv, cmap=plt.cm.gray,vmax=imagefv.max(),vmin=imagefv.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    imagefh=skimage.filters.farid_h(image) 
    
    plt.figure(figsize=(15, 7))
    plt.imshow(imagefh, cmap=plt.cm.gray,vmax=imagefh.max(),vmin=imagefh.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    imagef=np.add(imagefh,imagefv)
    plt.figure(figsize=(15, 7))
    plt.imshow(imagef, cmap=plt.cm.gray,vmax=imagef.max(),vmin=imagef.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений




    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




