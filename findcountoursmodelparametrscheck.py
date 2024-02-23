import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def genimage():
    N=int(input('Введите высоту изображения='))
    M=int(input('Введите ширину изображения='))
    x1=int(input('Введите абциссу круга 1='))
    y1=int(input('Введите ординату круга 1='))
    x2=int(input('Введите абциссу круга  2='))
    y2=int(input('Введите ординату круга 2='))
    x3=int(input('Введите абциссу круга  3='))
    y3=int(input('Введите ординату круга 3='))
    
    R1=int(input('Введите радиус 1='))
    R2=int(input('Введите радиус 2='))
    R3=int(input('Введите радиус 3='))
    image=np.zeros((M,N,3))
    print(image.shape)

    
    image=np.asarray(image,dtype=np.uint8)
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    imagem=cv2.circle(image, (x1, y1), R1, (255, 255, 255), -1)
    imagem=cv2.circle(imagem, (x2, y2), R2, (255, 255, 255), -1)
    imagem=cv2.circle(imagem, (x3, y3), R3, (255, 255, 255), -1)

    plt.figure(figsize=(15, 7))
    plt.imshow(imagem, cmap=plt.cm.gray,vmax=imagem.max(),vmin=imagem.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    return imagem,x1,x2,x3,y1,y2,y3,R1,R2,R3

def findcountour(image):
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    'поиск контуров'
    edges = cv2.Canny(image=image, threshold1=1, threshold2=255)
   
    plt.figure(figsize=(15,7))
    plt.imshow(edges,cmap='gray',vmax=edges.max(),vmin=edges.min())
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
    angleс=[]
    
    ret, thresh = cv2.threshold(edges, 1, 255,  cv2.THRESH_BINARY)
    plt.figure(figsize=(15,7))
    plt.imshow(thresh,cmap='gray',vmax=thresh.max(),vmin=thresh.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    k=0
    for num,i in enumerate(contours):
        M = cv2.moments(i)
        (x,y), (width, height), angle= cv2.minAreaRect(i)
        xi,yi,wi,hi = cv2.boundingRect(i)
        #rect = cv2.minAreaRect(i)
        #box = cv2.BoxPoints(rect)
        #print()
        #print(box)
        #box = np.int0(box)
        #print(box)
        #print()
       
        print(x)
        print(y)
        print()
        print(xi)
        print(yi)
        #print(width)
        #print(height)
        #print(angle)
        xcentrrect.append(x)
        ycentrrect.append(y)
        widthc.append(width)
        heightc.append(height) 
        angleс.append(angle)
        intensivityrect.append(image[int(np.floor(y)),int(np.floor(x))])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xcentr.append(cx)
            ycentr.append(cy)
            intensivity.append(image[cy,cx])
            k+=1
    print(f'Количество={k}')
   
    return xcentr,ycentr,xcentrrect,ycentrrect, widthc, heightc,angleс

def main():
    imagem,x1,x2,x3,y1,y2,y3,R1,R2,R3=genimage()
    xcentr,ycentr,xcentrrect,ycentrrect, widthc, heightc,anglec=findcountour(imagem)


    plt.show()
     
    
        
    
    
if __name__ == "__main__":
    main()

