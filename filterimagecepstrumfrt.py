#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import functools as ftools

from logzero import logger

try:
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
except: 
    from numpy.fft import fftshift, fftn, ifftn


def fftn_n( arr ):
    return fftn( arr, norm='ortho' )

def ifftn_n( arr ):
    return ifftn( arr, norm='ortho' )


chirp = np.mgrid[ 0:1, 0:1, 0:1 ]
chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )

pref0 = 'chirp = tuple( fftshift( this )**2 / this.shape[n] for n, this in enumerate( np.mgrid[ '
suff0 = ' ] ) )'

DoNothing = lambda x: x
opdict = { 0:DoNothing, 1:fftn_n, 2:np.flip, 3:ifftn_n }

def frft2( arr, alpha ):
    if arr.shape != chirp[0].shape:
        RecalculateChirp( arr.shape )
    ops = CanonicalOps( alpha )
    return frft_base( ops[0]( arr ), ops[1] )

def frft_base( arr, alpha ):
    phi = alpha * np.pi/2. 
    cotphi = 1. / np.tan( phi )
    cscphi = np.sqrt( 1. + cotphi**2 )
    scale = np.sqrt( 1. - 1.j*cotphi ) / np.sqrt( np.prod( arr.shape ) )
    modulator = ChirpFunction( cotphi - cscphi )
    filtor = ChirpFunction( cscphi )
    arr_frft = scale * modulator * ifftn_n( fftn_n( filtor ) * fftn_n( modulator * arr ) )
    return arr_frft

def ChirpFunction( x ):
    return np.exp( x * chirp_arg ) 

def RecalculateChirp( newshape ):
    logger.warning( 'Recalculating chirp. ' )
    global chirp_arg
    if len( newshape ) == 1:    # extra-annoying string manipulations needed with 1D data
        pref = pref0.replace( 'np.', '( np.' )
        suff = suff0.replace( ']', '], )' )
    else: 
        pref = pref0
        suff = suff0
    regrid = ','.join( tuple( '-%d:%d'%(n//2,n//2) for n in newshape ) ).join( [ pref, suff ] )
    #print( regrid )
    exec( regrid, globals() )
    chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )
    return

def CanonicalOps( alpha ):
    alpha_0 = alpha % 4. 
    if alpha_0 < 0.5:
        return[ ifftn_n, 1.+alpha_0 ]
    flag = 0
    while alpha_0 > 1.5:
        alpha_0 -= 1.
        flag += 1
    return [ opdict[flag], alpha_0 ]


"""
Module to calculate the fast fractional fourier transform.

"""

from __future__ import division
import numpy
import scipy
import scipy.signal


def frft(f, a):
    """
    Calculate the fast fractional fourier transform.

    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power

    Returns
    -------
    data : numpy array
        The transformed signal.


    References
    ---------
     .. [1] This algorithm implements `frft.m` from
        https://nalag.cs.kuleuven.be/research/software/FRFT/

    """
    ret = numpy.zeros_like(f, dtype=numpy.complex)
    f = f.copy().astype(numpy.complex)
    N = len(f)
    shft = numpy.fmod(numpy.arange(N) + numpy.fix(N / 2), N).astype(int)
    sN = numpy.sqrt(N)
    a = numpy.remainder(a, 4.0)

    # Special cases
    if a == 0.0:
        return f
    if a == 2.0:
        return numpy.flipud(f)
    if a == 1.0:
        ret[shft] = numpy.fft.fft(f[shft]) / sN
        return ret
    if a == 3.0:
        ret[shft] = numpy.fft.ifft(f[shft]) * sN
        return ret

    # reduce to interval 0.5 < a < 1.5
    if a >= 2.0:
        a = a - 2.0
        f = numpy.flipud(f)
    if a >= 1.5:
        a = a - 1
        f[shft] = numpy.fft.fft(f[shft]) / sN
    if a <= 0.5:
        a = a + 1
        f[shft] = numpy.fft.ifft(f[shft]) * sN

    # the general case for 0.5 < a < 1.5
    alpha = a * numpy.pi / 2
    tana2 = numpy.tan(alpha / 2)
    sina = numpy.sin(alpha)
    f = numpy.hstack((numpy.zeros(N - 1), sincinterp(f), numpy.zeros(N - 1))).T

    # chirp premultiplication
    chrp = numpy.exp(-1j * numpy.pi / N * tana2 / 4 *
                     numpy.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # chirp convolution
    c = numpy.pi / N / sina / 4
    ret = scipy.signal.fftconvolve(
        numpy.exp(1j * c * numpy.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8 * N - 7] * numpy.sqrt(c / numpy.pi)

    # chirp post multiplication
    ret = chrp * ret

    # normalizing constant
    ret = numpy.exp(-1j * (1 - a) * numpy.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def ifrft(f, a):
    """
    Calculate the inverse fast fractional fourier transform.

    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power

    Returns
    -------
    data : numpy array
        The transformed signal.

    """
    return frft(f, -a)


def sincinterp(x):
    N = len(x)
    y = numpy.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = scipy.signal.fftconvolve(
        y[:2 * N],
        numpy.sinc(numpy.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]


def filterimagecepstrumfrt(image):
    dataFT=frft2(image,0.5)
    image=dataFT.real
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    image=dataFT.imag
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
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


def main():
    image=readimage("C:/Users/evgen/Downloads/photo_2023-10-09_21-50-34.jpg")
    plt.figure(figsize=(15, 7))
    plt.imshow(image, cmap=plt.cm.gray,vmax=image.max(),vmin=image.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    filterimagecepstrumfrt(image)
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




