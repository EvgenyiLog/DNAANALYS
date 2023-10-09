#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy
from numpy.fft import fft, ifft
from matplotlib import pyplot as plt

def xcorr(x):
    """FFT based autocorrelation function, which is faster than numpy.correlate"""
     # x is supposed to be an array of sequences, of shape (totalelements, length)
    fftx = fft(x, n=(length*2-1), axis=1)
    ret = ifft(fftx * np.conjugate(fftx), axis=1)
    ret = fftshift(ret, axes=1)                                
    return ret
from itertools import product 
from numpy import empty, roll


def autocorrelate(x):
    """
    Compute the multidimensional autocorrelation of an nd array.
    input: an nd array of floats
    output: an nd array of autocorrelations
    """

    # used for transposes
    t = roll(range(x.ndim), 1)

    # pairs of indexes
    # the first is for the autocorrelation array
    # the second is the shift
    ii = [list(enumerate(range(1, s - 1))) for s in x.shape]

    # initialize the resulting autocorrelation array
    acor = empty(shape=[len(s0) for s0 in ii])

    # iterate over all combinations of directional shifts
    for i in product(*ii):
        # extract the indexes for
        # the autocorrelation array 
        # and original array respectively
        i1, i2 = asarray(i).T

        x1 = x.copy()
        x2 = x.copy()

        for i0 in i2:
            # clip the unshifted array at the end
            x1 = x1[:-i0]
            # and the shifted array at the beginning
            x2 = x2[i0:]

            # prepare to do the same for 
            # the next axis
            x1 = x1.transpose(t)
            x2 = x2.transpose(t)

        # normalize shifted and unshifted arrays
        x1 -= x1.mean()
        x1 /= x1.std()
        x2 -= x2.mean()
        x2 /= x2.std()

        # compute the autocorrelation directly
        # from the definition
        acor[tuple(i1)] = (x1 * x2).mean()

    return acor
from skimage.morphology import local_maxima
import numpy  as np
def rdf(data):
    h, w = data.shape[:2]
    print(h//2)
    print(w//2)
    dataFT = fft(data, axis=1)
    dataAC = ifft(dataFT * numpy.conjugate(dataFT), axis=1).real
    padding = numpy.zeros((data.shape[0], 3))
    dataPadded = numpy.concatenate((data, padding), axis=1)
    dataFT = fft(dataPadded, axis=1)
    dataAC = ifft(dataFT * numpy.conjugate(dataFT), axis=1).real
    x,y=local_maxima(dataAC, indices=True)
    corr=[]
    for i,j in zip(x, y):
        corr.append(dataAC[i,j])
    r=np.sqrt(np.add(np.square(np.subtract(x,h//2)),np.square(np.subtract(y,w//2))))
    #r=np.sqrt(np.add(np.square(x),np.square(y)))
    print(np.amax(r))
    print(np.amin(r))
    plt.figure(figsize=(15, 7))
    plt.plot(r,corr)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.grid(True)
    
    plt.figure(figsize=(15, 7))
    plt.plot(r,corr)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.grid(True)
    plt.xlim(0,50)
    
    
    
import cv2    
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

def main():
    image=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg") 
    rdf(image)
    plt.show()
    
    
if __name__ == "__main__":
    main()
    


# In[ ]:




