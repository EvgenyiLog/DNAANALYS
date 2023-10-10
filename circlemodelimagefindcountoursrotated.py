#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from PIL import Image
import opensimplex as simplex
from skimage.morphology import local_maxima
from skimage.feature import peak_local_max


import cv2
import numpy as np


def img_rectangle_cut(img, rect=None, angle=None):
    """Translate an image, defined by a rectangle. The image is cropped to the size of the rectangle
    and the cropped image can be rotated.
    The rect must be of the from (tuple(center_xy), tuple(width_xy), angle).
    The angle are in degrees.
    PARAMETER
    ---------
    img: ndarray
    rect: tuple, optional
        define the region of interest. If None, it takes the whole picture
    angle: float, optional
        angle of the output image in respect to the rectangle.
        I.e. angle=0 will return an image where the rectangle is parallel to the image array axes
        If None, no rotation is applied.
    RETURNS
    -------
    img_return: ndarray
    rect_return: tuple
        the rectangle in the returned image
    t_matrix: ndarray
        the translation matrix
    """
    if rect is None:
        if angle is None:
            angle = 0
        rect = (tuple(np.array(img.shape) * .5), img.shape, 0)
    box = cv2.boxPoints(rect)

    rect_target = rect_rotate(rect, angle=angle)
    pts_target = cv2.boxPoints(rect_target)

    # get max dimensions
    size_target = np.int0(np.ceil(np.max(pts_target, axis=0) - np.min(pts_target, axis=0)))

    # translation matrix
    t_matrix = cv2.getAffineTransform(box[:3].astype(np.float32),
                                      pts_target[:3].astype(np.float32))

    # cv2 needs the image transposed
    img_target = cv2.warpAffine(cv2.transpose(img), t_matrix, tuple(size_target))

    # undo transpose
    img_target = cv2.transpose(img_target)
    return img_target, rect_target, t_matrix


def reshape_cv(x, axis=-1):
    """openCV and numpy have a different array indexing (row, cols) vs (cols, rows), compensate it here."""
    if axis < 0:
        axis = len(x.shape) + axis
    return np.array(x).astype(np.float32)[(*[slice(None)] * axis, slice(None, None, -1))]

def connect(x):
    """Connect data for a polar or closed loop plot, i.e. np.append(x, [x[0]], axis=0)."""
    if isinstance(x, np.ma.MaskedArray):
        return np.ma.append(x, [x[0]], axis=0)
    else:
        return np.append(x, [x[0]], axis=0)


def transform_np(x, t_matrix):
    """Apply a transform on a openCV indexed array and return a numpy indexed array."""
    return transform_cv2np(reshape_cv(x), t_matrix)


def transform_cv2np(x, t_matrix):
    """Apply a transform on a numpy indexed array and return a numpy indexed array."""
    return reshape_cv(cv2.transform(np.array([x]).astype(np.float32), t_matrix)[0])


def rect_scale_pad(rect, scale=1., pad=40.):
    """Scale and/or pad a rectangle."""
    return (rect[0],
            tuple((np.array(rect[1]) + pad) * scale),
            rect[2])


def rect_rotate(rect, angle=None):
    """Rotate a rectangle by an angle in respect to it's center.
    The rect must be of the from (tuple(center_xy), tuple(width_xy), angle).
    The angle is in degrees.
    """
    if angle is None:
        angle = rect[2]
    rad = np.deg2rad(np.abs(angle))
    rot_matrix_2d = np.array([[np.cos(rad), np.sin(rad)],
                              [np.sin(rad), np.cos(rad)]])

    # cal. center of rectangle
    center = np.sum(np.array(rect[1]).reshape(1, -1) * rot_matrix_2d, axis=-1) * .5
    center = np.abs(center)

    return tuple(center), rect[1], angle


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
    
    
    imagem=cv2.normalize(imagem, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edges = cv2.Canny(imagem, 0, 255,1)
    plt.figure(figsize=(15,7))
    plt.imshow(edges,cmap='gray',vmax=edges.max(),vmin=edges.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    ret, thresh = cv2.threshold(edges, 1, 255, 0)
    print(ret)
    plt.figure(figsize=(15,7))
    plt.imshow(thresh,cmap='gray',vmax=thresh.max(),vmin=thresh.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(K1+K2+K3+K4+K5)
    print(len(contours)/(K1+K2+K3+K4+K5))
    # Get Box
    rect = cv2.minAreaRect(np.argwhere(imagem))

     # Apply transformation
    rect_scale = rect_scale_pad(rect, scale = 1., pad = 40.)
    img_return, rect_target, t_matrix = img_rectangle_cut(
    imagem, 
    rect_scale, 
    angle=0
    )
    # PLOT
    fig, ax = plt.subplots(ncols=2, figsize=(15,7      ))
    ax = ax.flatten()
    ax[0].imshow(imagem  ,cmap='gray')

    box_i = reshape_cv(cv2.boxPoints(rect))
    ax[0].plot(*connect(box_i).T, 'o-', color='gray', alpha=.75, label='Original Box')
    box_i = reshape_cv(cv2.boxPoints(rect_scale))
    ax[0].plot(*connect(box_i).T, 'o-', color='green', alpha=.75, label='Scaled Box')
    


    ax[1].imshow(img_return,cmap='gray')
    box_i = transform_cv2np(cv2.boxPoints(rect), t_matrix)
    ax[1].plot(*connect(box_i).T, 'o-', color='gray', alpha=.75, label='Original Box')

    

    ax[0].set_title('Original')
    ax[1].set_title('Translated')

    for axi in ax:
        axi.legend(loc=1)
    
    plt.tight_layout()
    
    
    
    
def main():
    imagem=modelimage()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




