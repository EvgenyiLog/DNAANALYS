#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import matplotlib.pyplot as plt
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
    
import math
class DCT(object):
    """
    This class DCT implements all the procedures for transforming a given 2D digital image
    into its corresponding frequency-domain image (Forward DCT Transform)
    """
    
    @classmethod
    def __computeSinglePoint2DCT(self, imge, u, v, N):
        """
        A private method that computes a single value of the 2D-DCT from a given image.

        Parameters
        ----------
        imge : ndarray
            The input image.
        
        u : ndarray
            The index in x-dimension.
            
        v : ndarray
            The index in y-dimension.

        N : int
            Size of the image.
            
        Returns
        -------
        result : float
            The computed single value of the DCT.
        """
        result = 0

        for x in range(N):
            for y in range(N):
                result += imge[x, y] * math.cos(((2*x + 1)*u*math.pi)/(2*N)) * math.cos(((2*y + 1)*v*math.pi)/(2*N))

        #Add the tau value to the result
        if (u==0) and (v==0):
            result = result/N
        elif (u==0) or (v==0):
            result = (math.sqrt(2.0)*result)/N
        else:
            result = (2.0*result)/N

        return result
    
    @classmethod
    def __computeSinglePointInverse2DCT(self, dctImge, x, y, N):
        """
        A private method that computes a single value of the 2D-DCT from a given image.

        Parameters
        ----------
        imge : ndarray
            The input image.
        
        u : ndarray
            The index in x-dimension.
            
        v : ndarray
            The index in y-dimension.

        N : int
            Size of the image.
            
        Returns
        -------
        result : float
            The computed single value of the DCT.
        """
        result = 0

        for u in range(N):
            for v in range(N):
                if (u==0) and (v==0):
                    tau = 1.0/N
                elif (u==0) or (v==0):
                    tau = math.sqrt(2.0)/N
                else:
                    tau = 2.0/N            
                result += tau * dctImge[u, v] * math.cos(((2*x + 1)*u*math.pi)/(2*N)) * math.cos(((2*y + 1)*v*math.pi)/(2*N))

        return result
    
    @classmethod
    def computeForward2DDCT(self, imge):
        """
        Computes/generates the 2D DCT of an input image in spatial domain.

        Parameters
        ----------
        imge : ndarray
            The input image to be transformed.

        Returns
        -------
        final2DDFT : ndarray
            The transformed image.
        """
 
        # Assuming a square image
        N = imge.shape[0]
        final2DDCT = np.zeros([N, N], dtype=float)
        for u in range(N):
            for v in range(N):
                #Compute the DCT value for each cells/points in the resulting transformed image.
                final2DDCT[u, v] = DCT.__computeSinglePoint2DCT(imge, u, v, N)
        return final2DDCT
    
    @classmethod
    def computeInverse2DDCT(self, imge):
        """
        Computes/generates the 2D DCT of an input image in spatial domain.

        Parameters
        ----------
        imge : ndarray
            The input image to be transformed.

        Returns
        -------
        final2DDFT : ndarray
            The transformed image.
        """
 
        # Assuming a square image
        N = imge.shape[0]
        finalInverse2DDCT = np.zeros([N, N], dtype=float)
        for x in range(N):
            for y in range(N):
                #Compute the DCT value for each cells/points in the resulting transformed image.
                finalInverse2DDCT[x, y] = DCT.__computeSinglePointInverse2DCT(imge, x, y, N)
        return finalInverse2DDCT
    
    @classmethod
    def normalize2DDCTByLog(self, dctImge):
        """
        Computes the log transformation of the transformed DCT image to make the range
        of the DCT values b/n 0 to 255
        
        Parameters
        ----------
        dctImge : ndarray
            The input DCT transformed image.

        Returns
        -------
        dctNormImge : ndarray
            The normalized version of the transformed image.
        """
        
        #Normalize the DCT values of a transformed image:
        dctImge = np.absolute(dctImge)
        dctNormImge = (255/ math.log10(255)) * np.log10(1 + (255/(np.max(dctImge))*dctImge))
        
        return dctNormImge


def main():
    image=modelimage()
    dctImage = DCT.computeForward2DDCT(image)
    dctNormImage = DCT.normalize2DDCTByLog(dctImage)
    fig, axarr = plt.subplots(2, 3, figsize=(10, 10))
    axarr[0][0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    axarr[0][0].set_title('Original Image')
    axarr[0][0].tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    axarr[0][1].imshow(np.absolute(dctImage), cmap=plt.get_cmap('gray'))
    axarr[0][1].set_title('Unnormalized DCT')
    axarr[0][1].tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений

    axarr[0][2].imshow(dctNormImage, cmap=plt.get_cmap('gray'))
    axarr[0][2].set_title('Normalized DCT')
    axarr[0][2].tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    inverseImage = np.round(DCT.computeInverse2DDCT(dctImage))
    plt.figure(figsize=(15,7))
    plt.imshow(inverseImage, cmap=plt.get_cmap('gray'))
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()





# In[ ]:




