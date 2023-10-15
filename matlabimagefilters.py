#!/usr/bin/env python
# coding: utf-8

# In[16]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from PIL import Image
import opensimplex as simplex
from skimage.morphology import local_maxima
from skimage.feature import peak_local_max
from scipy.special import iv, erf, erfinv
import numpy as np
from scipy.signal import gaussian

from matplotlib import patches

from math import ceil 
import pywt
import matlab
import matlab.engine

"""
ewt_params()
Parameter struct for empirical wavelet. Also sets defaults for each value.
Parameters are as follow:
    log                 - whether or not you take log of spectrum
    removeTrends        - Removes trends before finding boundaries
    spectrumRegularize  - regularizes spectrum before finding boundaries
    lengthFilter        - length for filters used in spectrum regularization
    sigmaFilter         - sigma for gaussian filter used in spectrum regularization
    typeDetect          - type of thresholding method for scale-space detection
    option              - Curvelet option for 2D Curvelet EWT
    N
    detect
    init_bounds
Author: Basile Hurat, Jerome Gilles"""
class ewt_params:
    def __init__(self, log=0, removeTrends='none', degree=2, spectrumRegularize='none', lengthFilter=7, sigmaFilter=2, N=10, detect='scalespace', typeDetect='otsu', option=1, init_bounds=[4, 8, 13, 30], t=1, n=6, niter=4, includeCenter=0, edges=0, real=1, tau=0.1):
        self.log = log
        self.removeTrends = removeTrends
        self.degree = degree
        self.spectrumRegularize = spectrumRegularize
        self.lengthFilter = lengthFilter
        self.sigmaFilter = sigmaFilter
        self.N = N
        self.detect = detect
        self.typeDetect = typeDetect
        self.option = option
        self.init_bounds = init_bounds
        self.t = t
        self.n = n
        self.niter = niter
        self.includeCenter = includeCenter
        self.edges = edges
        self.real = real
        self.tau = tau
    
"""
spectrumRegularize(f,params)
pre-processes spectrum before boundary detection by regularizing spectrum
Options include:
    gaussian    - gaussian smoothing: lengthFilter defines size of filter and 
                sigmaFilter defines std dev of gaussian
    average     - box filter smoothing: lengthFilter defines size of filter
    closing     - compute the upper envelope via a morphological closing 
                operator: lengthFilter defines size of filter
Input:
    f       - spectrum to regularize
    params  - parameters for EWT (see utilities) Used for choice of 
            regularization and details
Output:
    f2      - regularized spectrum
Author: Basile Hurat, Jerome Gilles"""
def spectrumRegularize(f, params):
    if params.spectrumRegularize.lower() == 'gaussian': #gaussian
        f2 = np.pad(f,params.lengthFilter // 2, 'reflect')
        Reg_Filter = gaussian(params.lengthFilter, params.sigmaFilter)
        Reg_Filter = Reg_Filter / sum(Reg_Filter)
        f2 = np.convolve(f2, Reg_Filter, mode='same')
        return f2[params.lengthFilter // 2:-params.lengthFilter // 2]
    elif params.spectrumRegularize.lower() == 'average': #average
        f2 = np.pad(f,params.lengthFilter // 2, 'reflect')
        Reg_Filter = np.ones(params.lengthFilter)
        Reg_Filter = Reg_Filter / sum(Reg_Filter)
        f2 = np.convolve(f2, Reg_Filter, mode='same')
        return f2[params.lengthFilter//2:-params.lengthFilter//2]
    elif params.spectrumRegularize.lower() == 'closing': #closing
        f2 = np.zeros(len(f))
        for i in range(0, len(f)):
            f2[i] = np.min(f[max(0 , i - params.lengthFilter):min(len(f) - 1, i + params.lengthFilter + 1)])
        return f2
    
def removeTrends(f, params):
    #still needs to be implemented
    if params.removeTrends.lower() == '    ':
        f = f / np.max(f)
        lw = np.log(np.arange(1, len(f) + 1))
        s = -np.sum(lw * np.log(f)) / sum(lw ** 2)
        f2 = f - np.arange(1, len(f) + 1) ** (-s)

    elif params.removeTrends.lower() == 'poly':
        p = np.polyfit(np.arange(0, len(f)), f,params.degree)
        f2 = f - np.polyval(p, np.arange(0, len(f)))
    
    elif params.removeTrends.lower() == 'morpho':
        locmax = localmin(-f)
        sizeEl = len(f)
        n = 1
        nplus = 1
        while n < len(f):
            if locmax[n] == 1:
                if sizeEl > (n - nplus):
                    sizeEl = n - nplus
                nplus = n
                n += 1
            n += 1
        f2 = f - (ewt_closing(f,sizeEl + 1) + ewt_opening(f,sizeEl + 1)) / 2

    elif params.removeTrends.lower() == 'tophat':
        locmax = localmin(-f)
        sizeEl = len(f)
        n = 1
        nplus = 1
        while n < len(f):
            if locmax[n] == 1:
                if sizeEl > (n - nplus):
                    sizeEl = n - nplus
                nplus = n
                n += 1
            n += 1
        f2 = f - ewt_opening(f, sizeEl + 1)

    elif params.removeTrends.lower() == 'opening':
        locmax = localmin(-f)
        sizeEl = len(f)
        n = 1
        nplus = 1
        while n < len(f):
            if locmax[n] == 1:
                if sizeEl > (n - nplus):
                    sizeEl = n - nplus
                nplus = n
                n += 1
            n += 1
        f2 = ewt_opening(f, sizeEl + 1)
    return f2
    
def ewt_opening(f, sizeEl):
    ope = ewt_dilation(ewt_erosion(f, sizeEl), sizeEl)
    return ope

def ewt_closing(f, sizeEl):
    clo = ewt_erosion(ewt_dilation(f, sizeEl), sizeEl)
    return clo

def ewt_erosion(f, sizeEl):
    s = np.copy(f)
    for x in range(0, len(f)):
        s[x] = np.min(f[max(0, x - sizeEl):min(len(f), x + sizeEl)])
    return s

def ewt_dilation(f, sizeEl):
    s = np.copy(f)
    for x in range(0, len(f)):
        s[x] = np.max(f[max(0, x - sizeEl):min(len(f), x + sizeEl)])
    return s

"""
showewt1dBoundaries(f,bounds)
Plots boundaries of 1D EWT on top of magnitude spectrum of the signal
Input:
    f       - original signal
    bounds  - detected bounds
Author: Basile Hurat, Jerome Gilles"""
def showewt1dBoundaries(f, bounds):
    ff = np.abs(np.fft.fft(f))
    h = np.max(ff)
    plt.figure()
    plt.suptitle('1D EWT Boundaries')
    plt.plot(
        np.arange(0, np.pi + 1 / (len(ff) / 2), 
        np.pi / (len(ff) / 2)), ff[0:len(ff) // 2 + 1]
        )
    for i in range(0, len(bounds)):
        plt.plot([bounds[i], bounds[i]], [0, h - 1], 'r--')
    plt.show()

"""
showTensorBoundaries(f,bounds_row,bounds_col)
Plots boundaries of 2D tensor EWT on top of magnitude spectrum of image
Input:
    f           - original image
    bounds_row  - detected bounds on rows
    bounds_col  - detected bounds on columns
Author: Basile Hurat, Jerome Gilles"""
def show2DTensorBoundaries(f, bounds_row, bounds_col):
    [h, w] = f.shape
    ff = np.fft.fft2(f)
    fig = plt.figure()
    plt.suptitle('2D EWT Tensor Boundaries')
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(np.log(np.abs(np.fft.fftshift(ff))), cmap='gray')
    
    #draw horizontal lines
    for i in range(0, len(bounds_row)):
        scaled_bound = bounds_row[i] * h / np.pi / 2
        plt.plot([h // 2 + scaled_bound, h // 2 + scaled_bound],[0, w - 1], 'r-')
        plt.plot([h // 2 - scaled_bound, h // 2 - scaled_bound],[0, w - 1], 'r-')
    #draw vertical lines
    for i in range(0, len(bounds_col)):
        scaled_bound = bounds_col[i] * w / np.pi / 2
        plt.plot([0, h - 1], [w // 2 + scaled_bound, w // 2 + scaled_bound], 'r-')
        plt.plot([0, h - 1], [w // 2 - scaled_bound, w // 2 - scaled_bound], 'r-')
    plt.show()

def show2DLPBoundaries(f, bounds_scales):
    [h, w] = f.shape
    ff = np.fft.fft2(f)
    fig = plt.figure()
    plt.suptitle('2D EWT Littlewood-Paley or Ridgelet Boundaries')
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(np.log(np.abs(np.fft.fftshift(ff))), cmap='gray')
    
    #plot scale bounds
    for i in range(0, len(bounds_scales)):
        rad = bounds_scales[i] * h / np.pi / 2
        circ = plt.Circle((h // 2 + 1, w // 2 + 1), rad, color='r', fill=0)
        ax.add_patch(circ)
    plt.show()

"""
showCurveletBoundaries(f,option,bounds_scales,bounds_angles)
Plots boundaries of 2D curvelet EWT on top of magnitude spectrum of image
Input:
    f           - original image
    option      - option for Curvelet
    bounds_row  - detected bounds on scales
    bounds_col  - detected bounds on angles
Author: Basile Hurat, Jerome Gilles"""
def show2DCurveletBoundaries(f, option, bounds_scales, bounds_angles):
    [h, w] = f.shape
    ff = np.fft.fft2(f)
    fig = plt.figure()
    plt.suptitle('2D EWT Curvelet Boundaries')
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(np.log(np.abs(np.fft.fftshift(ff))), cmap='gray')
    if option == 1: #scales and angles detected separately
        #first plot scale bounds
        for i in range(0, len(bounds_scales)):
            rad = bounds_scales[i] * h / np.pi / 2
            circ = plt.Circle((h // 2 + 1,w // 2 + 1), rad, color='r', fill=0)
            ax.add_patch(circ)
        #Then plot the angle bounds
        for i in range(0, len(bounds_angles)):
            if abs(bounds_angles[i]) < np.pi / 4: 
                #Do first half of line
                x0 = (1 + bounds_scales[0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y0 = (1 + bounds_scales[0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x1 = w - 1
                y1 = (h + w * np.tan(bounds_angles[i])) // 2
                plt.plot([x0, x1], [y0, y1], 'r-')
                #Do second half of line
                x2 = (1 - bounds_scales[0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y2 = (1 - bounds_scales[0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x3 = 0
                y3 = (h - w * np.tan(bounds_angles[i])) // 2
                plt.plot([x2, x3], [y2, y3], 'r-')
            else:
                x0 = (1 - bounds_scales[0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y0 = (1 - bounds_scales[0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x1 = (w + h / np.tan(bounds_angles[i])) // 2
                y1 = h - 1
                plt.plot([x0, x1], [y0, y1], 'r-')
                x2 = (1 + bounds_scales[0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y2 = (1 + bounds_scales[0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x3 = (h - w / np.tan(bounds_angles[i])) // 2
                y3 = 0
                plt.plot([x2, x3], [y2, y3], 'r-')
                
    elif option == 2: #scales detected first and angles detected per scale
        #first plot scale bounds
        for i in range(0, len(bounds_scales)):
            rad = bounds_scales[i] * h / np.pi / 2
            circ = plt.Circle((h // 2 + 1, w // 2 + 1),rad, color='r', fill=0)
            ax.add_patch(circ)
        #Then plot the angle bounds for each scale
        for i in range(0, len(bounds_scales) - 1):
            for j in range(0, len(bounds_angles[i])): 
                if abs(bounds_angles[i][j]) < np.pi / 4: 
                    #Do first half of line
                    x0 = (1 + bounds_scales[i] * np.cos(bounds_angles[i][j]) / np.pi) * (w // 2 + 1)
                    y0 = (1 + bounds_scales[i] * np.sin(bounds_angles[i][j]) / np.pi) * (h // 2 + 1)
                    x1 = (1 + bounds_scales[i + 1] * np.cos(bounds_angles[i][j]) / np.pi) * (w // 2 + 1)
                    y1 = (1 + bounds_scales[i + 1] * np.sin(bounds_angles[i][j]) / np.pi) * (h // 2 + 1)
                    plt.plot([x0, x1], [y0, y1], 'r-')
                    #Do second half of line
                    x2 = (1 - bounds_scales[i] * np.cos(bounds_angles[i][j]) / np.pi) *(w // 2 + 1)
                    y2 = (1 - bounds_scales[i] * np.sin(bounds_angles[i][j]) / np.pi) *(h // 2 + 1)
                    x3 = (1 - bounds_scales[i+1] * np.cos(bounds_angles[i][j]) / np.pi) *(w // 2 + 1)
                    y3 = (1 - bounds_scales[i+1] * np.sin(bounds_angles[i][j]) / np.pi) *(h // 2 + 1)
                    plt.plot([x2, x3], [y2, y3], 'r-')
                else:
                    x0 = (1 - bounds_scales[i] * np.cos(bounds_angles[i][j]) / np.pi) * (w // 2 + 1)
                    y0 = (1 - bounds_scales[i] * np.sin(bounds_angles[i][j]) / np.pi) * (h // 2 + 1)
                    x1 = (1 - bounds_scales[i + 1] * np.cos(bounds_angles[i][j]) / np.pi) * (w // 2 + 1)
                    y1 = (1 - bounds_scales[i + 1] * np.sin(bounds_angles[i][j]) / np.pi) * (h // 2 + 1)
                    plt.plot([x0, x1], [y0, y1], 'r-')
    
                    x2 = (1 + bounds_scales[i] * np.cos(bounds_angles[i][j]) / np.pi) * (w // 2 + 1)
                    y2 = (1 + bounds_scales[i] * np.sin(bounds_angles[i][j]) / np.pi) * (h // 2 + 1)
                    x3 = (1 + bounds_scales[i + 1] * np.cos(bounds_angles[i][j]) / np.pi) * (w // 2 + 1)
                    y3 = (1 + bounds_scales[i + 1] * np.sin(bounds_angles[i][j]) / np.pi) * (h // 2 + 1)
                    plt.plot([x2, x3], [y2, y3], 'r-')
        #Then take care of last scale
        for i in range(0, len(bounds_angles[-1])): 
            if abs(bounds_angles[-1][i]) < np.pi / 4: 
                #Do first half of line
                x0 = (1 + bounds_scales[-1] * np.cos(bounds_angles[-1][i]) / np.pi) * (w // 2 + 1)
                y0 = (1 + bounds_scales[-1] * np.sin(bounds_angles[-1][i]) / np.pi) * (h // 2 + 1)
                x1 = w - 1
                y1 = (h + w * np.tan(bounds_angles[-1][i])) // 2
                plt.plot([x0, x1], [y0, y1], 'r-')
                #Do second half of line
                x2 = (1 - bounds_scales[-1] * np.cos(bounds_angles[-1][i]) / np.pi) * (w // 2 + 1)
                y2 = (1 - bounds_scales[-1] * np.sin(bounds_angles[-1][i]) / np.pi) * (h // 2 + 1)
                x3 = 0
                y3 = (h - w * np.tan(bounds_angles[-1][i])) // 2
                plt.plot([x2, x3], [y2, y3], 'r-')
            else:
                x0 = (1 - bounds_scales[-1] * np.cos(bounds_angles[-1][i]) / np.pi) * (w // 2 + 1)
                y0 = (1 - bounds_scales[-1] * np.sin(bounds_angles[-1][i]) / np.pi) * (h // 2 + 1)
                x1 = (w + h / np.tan(bounds_angles[-1][i])) // 2
                y1 = h - 1
                plt.plot([x0, x1], [y0, y1], 'r-')
                x2 = (1 + bounds_scales[-1] * np.cos(bounds_angles[-1][i]) / np.pi) * (w // 2 + 1)
                y2 = (1 + bounds_scales[-1] * np.sin(bounds_angles[-1][i]) / np.pi) * (h // 2 + 1)
                x3 = (h - w / np.tan(bounds_angles[-1][i])) // 2
                y3 = 0
                plt.plot([x2, x3], [y2, y3], 'r-')
    
    elif option == 3: #angles detected first and scales detected per angle
        #plot first scale
        rad = bounds_scales[0][0] * h / np.pi / 2
        circ = plt.Circle((h // 2, w // 2), rad, color='r', fill=0)
        ax.add_patch(circ)
        
        #Plot angle bounds first
        for i in range(0, len(bounds_angles)): 
            if abs(bounds_angles[i]) < np.pi / 4: 
                #Do first half of line
                x0 = (1 + bounds_scales[0][0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y0 = (1 + bounds_scales[0][0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x1 = w - 1
                y1 = (h + w * np.tan(bounds_angles[i])) // 2
                plt.plot([x0, x1], [y0, y1], 'r-')
                #Do second half of line
                x2 = (1 - bounds_scales[0][0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y2 = (1 - bounds_scales[0][0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x3 = 0
                y3 = (h - w * np.tan(bounds_angles[i])) // 2
                plt.plot([x2, x3], [y2, y3], 'r-')
            else:
                x0 = (1 - bounds_scales[0][0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y0 = (1 - bounds_scales[0][0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x1 = (w + h / np.tan(bounds_angles[i])) // 2
                y1 = h - 1
                plt.plot([x0, x1], [y0, y1], 'r-')
                x2 = (1 + bounds_scales[0][0] * np.cos(bounds_angles[i]) / np.pi) * w // 2
                y2 = (1 + bounds_scales[0][0] * np.sin(bounds_angles[i]) / np.pi) * h // 2
                x3 = (h - w / np.tan(bounds_angles[i])) // 2
                y3 = 0
                plt.plot([x2, x3], [y2, y3], 'r-')
        #For each angular sector, plot arc for scale
        for i in range(0, len(bounds_angles) - 1): 
            for j in range(0, len(bounds_scales[i + 1])):
                rad = bounds_scales[i + 1][j] * h / np.pi
                arc = patches.Arc(
                    (h // 2, w // 2), 
                    rad, 
                    rad, 
                    0, 
                    bounds_angles[i] * 180 / np.pi,
                    bounds_angles[i + 1] * 180 / np.pi,
                    color='r',
                    Fill=0
                    )
                ax.add_patch(arc)
                arc2 = patches.Arc(
                    (h // 2, w // 2),
                    rad,
                    rad,
                    0,
                    180 + bounds_angles[i] * 180 / np.pi,
                    180 + bounds_angles[i + 1] * 180 / np.pi,
                    color='r',
                    Fill=0
                    )
                ax.add_patch(arc2)
        #Plot arcs for last angular sector
        for i in range(0, len(bounds_scales[-1])):
            rad = bounds_scales[-1][i] * h / np.pi
            arc = patches.Arc(
                (h // 2, w // 2),
                rad,
                rad,
                0,
                bounds_angles[-1] * 180 / np.pi,
                180 + bounds_angles[1] * 180 / np.pi,
                color='r',
                Fill=0
                )
            ax.add_patch(arc)
            arc2 = patches.Arc(
                (h // 2, w // 2),
                rad,
                rad,
                0,
                180 + bounds_angles[-1] * 180 / np.pi,
                360 + bounds_angles[1] * 180 / np.pi,
                color='r',
                Fill=0
                )
            ax.add_patch(arc2)
    else:
        return -1
    plt.show()
    
"""
showEWT1DCoefficients(ewtc)
Plots coefficients of the 1D ewt
Input:
    ewtc - 1D empirical wavelet coefficients gotten from the ewt1d function
Author: Basile Hurat, Jerome Gilles"""
def showEWT1DCoefficients(ewtc):
    if len(ewtc) < 10:
        fig = plt.figure()
        fig.suptitle("1D EWT coefficients")
        for i in range(0, len(ewtc)):
            plt.subplot(len(ewtc), 1, i + 1)
            plt.plot(ewtc[i])
        plt.show()
    else:
        for i in range(0, len(ewtc)):
            if i % 10 == 0:
                plt.figure()
            plt.subplot(10, 1, i % 10 + 1)
            plt.plot(ewtc[i])
        plt.show()
           
"""
showEWT2DCoefficients(ewtc)
Plots coefficients of the 2D empirical wavelet transform
Input:
    ewtc        - 2D empirical wavelet coefficients 
    ewt_type    - the transform used to get the empirical wavelet coefficent
    option      - (optional) the curvelet option, should you need to specify
Author: Basile Hurat, Jerome Gilles"""
def showEWT2DCoefficients(ewtc, ewt_type, option=1):
    if ewt_type.lower() == 'tensor':
        m = len(ewtc)
        n = len(ewtc[0])
        fig = plt.figure()
        fig.suptitle("Tensor EWT coefficients")
        for i in range(0, m):
            for j in range(0, n):
                plt.subplot(m, n, i * m + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(ewtc[i][j], cmap='gray')
                plt.xlabel(f'i = {i}, j = {j}')
        plt.show()
    if ewt_type.lower() == 'lp':
        m = len(ewtc)
        fig = plt.figure()
        fig.suptitle("Littlewood-Paley EWT coefficients")
        for i in range(0, m):
            plt.subplot(np.ceil(np.sqrt(m)), np.ceil(np.sqrt(m)), i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(ewtc[i], cmap='gray')
            plt.xlabel(f'i = {i}')
        plt.show()
    if ewt_type.lower() == 'ridgelet':
        m = len(ewtc)
        fig = plt.figure()
        fig.suptitle("Ridgelet EWT coefficients")
        for i in range(0, m):
            plt.subplot(np.ceil(np.sqrt(m)), np.ceil(np.sqrt(m)), i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(ewtc[i], cmap='gray')
            plt.xlabel(f'i = {i}')
        plt.show()
    if ewt_type.lower() == 'curvelet':
        fig = plt.figure()
        fig.suptitle("Curvelet EWT coefficient for scale 0")
        plt.imshow(ewtc[0][0], cmap='gray')
        if option < 3:
            for i in range(1, len(ewtc)):
                fig = plt.figure()
                fig.suptitle(f'Curvelet EWT coefficients for scale {i}')
                m = len(ewtc[i])
                for j in range(0, m):
                    plt.subplot(ceil(np.sqrt(m)), ceil(np.sqrt(m)), j + 1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(ewtc[i][j], cmap='gray')
                    plt.xlabel(f'Angle {j}')
            plt.show()
        else:
            for i in range(1, len(ewtc)):
                fig = plt.figure()
                fig.suptitle(f'Curvelet EWT coefficients for Angle {i}')
                m = len(ewtc[i])
                for j in range(0, m):
                    plt.subplot(np.ceil(np.sqrt(m)), np.ceil(np.sqrt(m)), j + 1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(ewtc[i][j], cmap='gray')
                    plt.xlabel(f'Scale {j}')
            plt.show()

"""
ewt1d(f,params)
Performs the 1D empirical wavelet transform
Input:
    f       - 1D signal
    params  - parameters for EWT (see utilities)
Output:
    ewt     - empirical wavelet coefficients
    mfb     - empirical wavelet filter bank
    bounds  - bounds detected on Fourier spectrum
Author: Basile Hurat, Jerome Gilles"""

def ewt1d(f, params):
    ff = np.fft.fft(f) 
    real = all(np.isreal(f)) #Check if function is real
    #performs boundary detection
    
    bounds = ewt_boundariesDetect(np.abs(ff), params, real)
    if real:
        bounds = bounds * np.pi / (np.round(len(ff) / 2))
    else:
        bounds = bounds * 2 * np.pi / len(ff)
    
    
    #From bounds, construct filter bank
    mfb = ewt_LP_Filterbank(bounds, len(ff), real)
    
    #Filter to get empirical wavelet coefficients
    ewt = []
    for i in range(0, len(mfb)):
        if real:
            ewt.append(np.real(np.fft.ifft(np.conj(mfb[i]) * ff)))
        else:
            ewt.append(np.fft.ifft(np.conj(mfb[i]) * ff))
    return [ewt, mfb, bounds]

"""
iewt1d(f,params)
Performs the inverse 1D empirical wavelet transform
Input:
    ewt     - empirical wavelet coefficients
    mfb     - empirical wavelet filter bank
Output:
    rec     - reconstruction of signal which generated empirical wavelet 
          coefficients and empirical wavelet filter bank
Author: Basile Hurat, Jerome Gilles"""

def iewt1d(ewt, mfb):
    real = all(np.isreal(ewt[0]))
    if real:
        rec = np.zeros(len(ewt[0]))
        for i in range(0, len(ewt)):
            rec += np.real(np.fft.ifft(np.fft.fft(ewt[i]) * mfb[i]))
    else:
        rec = np.zeros(len(ewt[0])) * 0j
        for i in range(0, len(ewt)):
            rec += np.fft.ifft(np.fft.fft(ewt[i]) * mfb[i])
    return rec

"""
ewt_LP_Filterbank(bounds,N)
Construct empirical wavelet filterbank based on a set of boundaries in the 
Fourier spectrum
Input:
    bounds  - detected bounds in the Fourier spectrum
    N       - desired size of filters (usually size of original signal)
    real    - flag for if original signal is real or complex
Output:
    mfb     - resulting empirical wavelet filter bank
Author: Basile Hurat, Jerome Gilles"""
def ewt_LP_Filterbank(bounds, N, real):
    #Calculate Gamma
    gamma = 1
    for i in range(0, len(bounds) - 1):
        r = (bounds[i + 1] - bounds[i]) / (bounds[i + 1] + bounds[i])
        if r < gamma and r > 1e-16:
            gamma = r
    if real:
        r = (np.pi - bounds[-1]) / (np.pi + bounds[-1])
    else:
        r = (2 * np.pi - bounds[-1])/(2 * np.pi + bounds[-1])
    if r < gamma:
        gamma = r
    gamma *= (1 - 1 / N) #ensures strict inequality
    if real == 0:
        num_bounds = len(bounds)
        i = 0
        while i < num_bounds:   # if difference between bound and pi less than.. 
            if num_bounds == 1:
                break
            if abs(bounds[i] - np.pi) < gamma: #gamma, remove
                bounds = np.delete(bounds, i)
                num_bounds -= 1
            else:
                i += 1
    
    aw = np.arange(0, 2 * np.pi - 1 / N, 2 * np.pi / N)
    if real == 1:
        aw[N // 2:] -= 2 * np.pi 
        aw = np.abs(aw)
        filterbank = []
        filterbank.append(ewt_LP_Scaling(bounds[0], aw, gamma, N))
        for i in range(1, len(bounds)):
            filterbank.append(ewt_LP_Wavelet(bounds[i - 1],bounds[i], aw, gamma, N))
        filterbank.append(ewt_LP_Wavelet(bounds[-1], np.pi, aw, gamma, N))
    else:
        filterbank = []
        filterbank.append(ewt_LP_Scaling_Complex(bounds[0], bounds[-1], aw, gamma, N))
        for i in range(0, len(bounds) - 1):
            if ((bounds[i] <= np.pi) and (bounds[i + 1] > np.pi)):
                filterbank.append(ewt_LP_Wavelet_ComplexLow(bounds[i], aw, gamma, N))
                filterbank.append(ewt_LP_Wavelet_ComplexHigh(bounds[i + 1], aw, gamma, N))
            else:
                filterbank.append(ewt_LP_Wavelet(bounds[i], bounds[i + 1], aw, gamma, N))
    return filterbank

"""
ewt_LP_Scaling(w1,aw,gamma,N)
Constructs empirical scaling function, which is the low-pass filter for EWT 
filterbank
Input:
    w1      - first bound, which delineates the end of low-pass filter
    aw      - reference vector which goes from 0 to 2pi
    gamma   - gamma value which guarantees tight frame
Output:
    yms     - resulting empirical scaling function
Author: Basile Hurat, Jerome Gilles"""
def ewt_LP_Scaling(w1, aw, gamma, N):
    mbn = (1 - gamma) * w1 #beginning of transition
    pbn = (1 + gamma) * w1 #end of transition
    an = 1 / (2 * gamma * w1) #scaling in beta function

    yms = 1.0 * (aw <= mbn) #if less than lower bound, equals 1
    yms += (aw > mbn) * (aw <= pbn) \
        * np.cos(np.pi * ewt_beta(an * (aw - mbn)) / 2) #Transition area
    return yms

"""
ewt_LP_Wavelet(wn, wm, aw, gamma, N)
Constructs empirical wavelet, which is a band-pass filter for EWT filterbank
Input:
    wn      - lower bound, which delineates the beginning of band-pass filter
    wm      - higher bound, which delineates the end of band-pass filter
    aw      - reference vector which goes from 0 to 2pi
    gamma   - gamma value which guarantees tight frame
Output:
    ymw     - resulting empirical wavelet 
Author: Basile Hurat, Jerome Gilles"""
def ewt_LP_Wavelet(wn, wm, aw, gamma, N):
    if wn > np.pi:  #If greater than pi, subtract 2pi, otherwise dont
        a = 1
    else:
        a = 0

    mbn = wn - gamma * abs(wn - a * 2 * np.pi) #beginning of first transition
    pbn = wn + gamma * abs(wn - a * 2 * np.pi) #end of first transition
    an = 1 / (2 * gamma * abs(wn - a * 2 * np.pi)) #scaling in first transition's beta function

    if wm > np.pi: #If greater than pi, subtract 2pi, otherwise dont
        a=1
    else:
        a=0
    
    mbm = wm - gamma * abs(wm - a * 2 * np.pi) #beginning of second transition
    pbm = wm + gamma * abs(wm - a * 2 * np.pi) #end of second transition
    am = 1 / (2 * gamma * abs(wm - a * 2 * np.pi))  #scaling in second transition's beta function
    
    ymw = 1.0 * (aw > mbn) * (aw< pbm) #equals 1 between transition areas
    case = (aw > mbn) * (aw < pbn)
    ymw[case] *= np.sin(np.pi * ewt_beta(an * (aw[case] - mbn)) / 2) #1st transition area
    if wm < np.pi:
        case = (aw > mbm) * (aw < pbm)
        ymw[case] *= np.cos(np.pi * ewt_beta(am * (aw[case] - mbm)) / 2) #2nd transition area
    return ymw

"""
ewt_LP_Scaling_Complex(wn,wm,aw,gamma,N)
Constructs assymmetrical scaling filter for complex wavelet with support 
[wn,wm]
Input:
    wn      - higher bound, which delineates the end of band-pass filter
    wm      - higher bound, which delineates the end of band-pass filter
    aw      - reference vector which goes from 0 to 2pi
    gamma   - gamma value which guarantees tight frame
Output:
    ymw     - resulting empirical wavelet 
Author: Basile Hurat, Jerome Gilles"""    
def ewt_LP_Scaling_Complex(wn, wm, aw, gamma, N):
    if wn == wm:
        return np.ones(N)
    
    if wn > np.pi:  #If greater than pi, subtract 2pi, otherwise dont
        a = 1
    else:
        a = 0

    mbn = wn - gamma * abs(wn - a * 2 * np.pi) #beginning of first transition
    pbn = wn + gamma * abs(wn - a * 2 * np.pi) #end of first transition
    an = 1 / (2 * gamma * abs(wn - a * 2 * np.pi)) #scaling in first transition's beta function

    if wm > np.pi: #If greater than pi, subtract 2pi, otherwise dont
        a=1
    else:
        a=0
    
    mbm = wm - gamma * abs(wm - a * 2 * np.pi) #beginning of second transition
    pbm = wm + gamma * abs(wm - a * 2 * np.pi) #end of second transition
    am = 1 / (2 * gamma * abs(wm - a * 2 * np.pi))  #scaling in second transition's beta function
    
    ymw = 1.0 * (aw <= mbn) + 1.0 * (aw >= pbm) #equals 1 between transition areas
    case = (aw >= mbn) * (aw <= pbn)
    ymw[case] = np.cos(np.pi * ewt_beta(an * (aw[case] - mbn)) / 2) #1nd transition area    
    case = (aw >= mbm) * (aw <= pbm)
    ymw[case] = np.sin(np.pi * ewt_beta(am * (aw[case] - mbm)) / 2) #2st transition area    
    return ymw

"""
ewt_LP_Wavelet_ComplexLow(wn,aw,gamma,N)
Constructs upper transition for complex wavelet with support [wn,pi]
Input:
    wn      - lower bound, which delineates the beginning of band-pass filter
    aw      - reference vector which goes from 0 to 2pi
    gamma   - gamma value which guarantees tight frame
Output:
    ymw     - resulting empirical wavelet 
Author: Basile Hurat, Jerome Gilles"""    
def ewt_LP_Wavelet_ComplexLow(wn, aw, gamma, N):
    if wn > np.pi:  #If greater than pi, subtract 2pi, otherwise dont
        a = 1
    else:
        a = 0
    
    an = 1 / (2 * gamma * abs(wn - a * 2 * np.pi)) #scaling in lowertransition's beta function
    mbn = wn - gamma * abs(wn - a * 2 * np.pi) #beginning of lower transition
    pbn = wn + gamma * abs(wn - a * 2 * np.pi) #end of lower transition
    
    ymw = 1.0 * (aw >= pbn) * (aw <= np.pi)
    case = (aw >= mbn) * (aw<=pbn)
    ymw[case] = np.sin(np.pi * ewt_beta(an * (aw[case] - mbn)) / 2) #lower transition area
    return ymw

"""
ewt_LP_Wavelet_ComplexHigh(wn,aw,gamma,N)
Constructs upper transition for complex wavelet with support [pi,wn]
Input:
    wn      - higher bound, which delineates the end of band-pass filter
    aw      - reference vector which goes from 0 to 2pi
    gamma   - gamma value which guarantees tight frame
Output:
    ymw     - resulting empirical wavelet 
Author: Basile Hurat, Jerome Gilles"""    
def ewt_LP_Wavelet_ComplexHigh(wn, aw, gamma, N):
    if wn > np.pi:  #If greater than pi, subtract 2pi, otherwise dont
        a = 1
    else:
        a = 0
    
    an = 1 / (2 * gamma * abs(wn - a * 2 * np.pi)) #scaling in upper transition's beta function
    mbn = wn - gamma * abs(wn - a * 2 * np.pi) #beginning of upper transition
    pbn = wn + gamma * abs(wn - a * 2 * np.pi) #end of upper transition
    
    ymw = 1.0 * (aw > np.pi) * (aw <= mbn)
    case = (aw >= mbn) * (aw <= pbn)
    ymw[case] = np.cos(np.pi * ewt_beta(an * (aw[case] - mbn)) / 2) #upper transition area
    return ymw

"""
ewt_beta(x)
Beta function that is used in empirical wavelet and empirical scaling function
construction
Input:
    x       - vector x as an input
Output:
    bm      - beta function applied to vector
Author: Basile Hurat, Jerome Gilles"""
def ewt_beta(x):
    bm = (x >= 0) * (x <= 1) \
        * (x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 *x **3))
    bm += (x > 1)
    return bm

"""
ewt_boundariesDetect(absf, params)
Adaptively detects boundaries in 1D magnitude Fourier spectrum based on the 
detection method chosen in params.detect
Input:
    absf    - magnitude Fourier spectrum
    params  - parameters for EWT (see utilities)
Output:
    bounds     - resulting boundaries in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_boundariesDetect(absf, params, sym = 1):
    if params.log == 1:     #apply log parameter
        preproc = np.log(absf)
    else:
        preproc = np.copy(absf)
    if params.removeTrends.lower() != 'none':   #apply removeTrend parameter
        preproc = removeTrends(absf, params)
    if params.spectrumRegularize.lower() != 'none': #apply spectrumRegularize parameter
        preproc = spectrumRegularize(preproc, params)
    
    #Choose detection method
    if params.detect == 'scalespace':
        bounds = ewt_GSSDetect(preproc, params, sym)
    elif params.detect == 'locmax':
        if sym == 1:
            bounds = ewt_localMaxBounds(preproc[0:len(preproc) // 2], params.N)
        else:
            bounds = ewt_localMaxBounds(preproc, params.N)
    elif params.detect == 'locmaxmin':
        if sym == 1:
            bounds = ewt_localMaxMinBounds(preproc[0:len(preproc) // 2], params.N)
        else:
            bounds = ewt_localMaxMinBounds(preproc, params.N)
    elif params.detect == 'locmaxminf':
        if sym == 1:
            bounds = ewt_localMaxMinBounds(preproc[0:len(preproc) // 2], params.N,absf[0:len(absf) // 2])
        else:
            bounds = ewt_localMaxMinBounds(preproc, params.N, absf)
    elif params.detect == 'adaptivereg':
        if sym == 1:
            bounds = ewt_adaptiveBounds(preproc[0:len(preproc) // 2], params.init_bounds)
        else:
            bounds = ewt_adaptiveBounds(preproc, params.init_bounds)        
    elif params.detect == 'adaptive':
        if sym == 1:
            bounds = ewt_adaptiveBounds(preproc[0:len(preproc) // 2], params.init_bounds,absf[0:len(absf) // 2])
        else:
            bounds = ewt_adaptiveBounds(preproc, params.init_bounds, absf)
    for i in range(0, len(bounds)):
        if bounds[i] == 0:
            bounds = np.delete(bounds, i)
            break
    return bounds

"""
ewt_localMaxBounds(f, N)
Detects N highest maxima, and returns the midpoints between them as detected 
boundaries
Input:
    f       - signal to detect maxima from (generally pre-processed magnitude spectrum)
    N       - number of maxima to detect
Output:
    bounds  - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_localMaxBounds(f, N):
    #Detect maxima
    maxima = localmin(-f).astype(bool)
    index = np.arange(0, len(maxima))
    maxindex = index[maxima]
    #If we have more than N, keep only N highest maxima values
    if N < len(maxindex):
        order = np.argsort(f[maxima])[-N:]  
        maxindex = np.sort(maxindex[order])
    else:
        N = len(maxindex) - 1
    #find midpoints
    bounds = np.zeros(N)
    bounds[0] = round(maxindex[0] / 2)
    for i in range(0, N-1):
        bounds[i + 1] = (maxindex[i] + maxindex[i + 1]) // 2
    return bounds

"""
ewt_localMaxMinBounds(f, N,f_orig)
Detects N highest maxima, and returns the lowest minima between them as detected 
boundaries
Input:
    f       - signal to detect maxima and minima from (generally pre-processed 
            magnitude spectrum)
    N       - number of maxima to detect
    f_orig  - (Optional) If given, detects minima from this instead of f
Output:
    bounds  - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_localMaxMinBounds(f, N, f_orig = []): 
    #Get both maxima and minima of signal
    maxima = localmin(-f).astype(bool)
    if len(f_orig) == 0:
        minima = localmin(f).astype(bool)
    else:
        minima = localmin(f_orig).astype(bool)
    index = np.arange(0, len(maxima))
    maxindex = index[maxima]
    minindex = index[minima]
    
    #If we have more than N, keep only N highest maxima values
    if N<len(maxindex):
        order = np.argsort(f[maxima])[-N:]  
        maxindex = np.sort(maxindex[order])
    else:
        N = len(maxindex) - 1
    
    bounds = np.zeros(N)
    intervalmin = minindex[minindex < maxindex[0]]
    if not len(intervalmin) == 0:
        bounds[0] = intervalmin[np.argmin(f[intervalmin])]
    
    for i in range(0,N - 1):
        intervalmin = minindex[minindex > maxindex[i]]
        intervalmin = intervalmin[intervalmin < maxindex[i + 1]]
        bounds[i + 1] = intervalmin[np.argmin(f[intervalmin])]
    return bounds

"""
ewt_adaptiveBounds(f, N,f_orig)
Adaptively detect from set of initial bounds. Returns lowest minima within a 
neighborhood of given bounds
Input:
    f               - signal to detect maxima and minima from (generally 
                    pre-processed magnitude spectrum)
    init_bounds0    - initial bounds to look at  detection
    f_orig          - (Optional) If given, detects minima from this instead of f
Output:
    bounds          - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_adaptiveBounds(f, init_bounds0, f_orig=[]):
    if len(f_orig) != 0:
        f = np.copy(f_orig)
    init_bounds = []
    init_bounds[:] = init_bounds0
    init_bounds.insert(0, 0)
    init_bounds.append(len(f))
    bounds = np.zeros(len(init_bounds) - 1)
    for i in range(0,len(init_bounds) - 1):
        neighb_low = round(init_bounds[i + 1] - round(abs(init_bounds[i + 1] - init_bounds[i])) / 2)
        neighb_high = round(init_bounds[i + 1] + round(abs(init_bounds[i + 1] - init_bounds[i])) / 2)
        bounds[i] = np.argmin(f[neighb_low:neighb_high + 1])
    return np.unique(bounds)

"""
ewt_GSSDetect(f, params,sym)
Detects boundaries using scale-space. 
Input:
    f       - signal to detect boundaries between
    params  - parameters for EWT (see utilities). Notably, the adaptive 
            threshold from params.typeDetect
    sym     - parameter whether or not the signal is symmetric. If true, 
            returns bounds less than middle index
Output:
    bounds  - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""  
def ewt_GSSDetect(f, params, sym):
    #Apply gaussian scale-space
    plane = GSS(f)
    #Get persistence (lengths) and indices of minima
    [lengths, indices] = lengthScaleCurve(plane)
    if sym == 1:
        lengths = lengths[indices < len(f) / 2 - 1] #Halve the spectrum
        indices= indices[indices < len(f) / 2 - 1] #Halve the spectrum    
    #apply chosen thresholding method
    if params.typeDetect.lower() == 'otsu':    
        thresh = otsu(lengths)
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'mean':   
        thresh = np.ceil(np.mean(lengths))
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'empiricallaw':
        thresh = empiricalLaw(lengths)
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'halfnormal':
        thresh = halfNormal(lengths)
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'kmeans':
        clusters = ewtkmeans(lengths, 1000)
        upper_cluster = clusters[lengths == max(lengths)][0]
        bounds = indices[clusters == upper_cluster]        

    return bounds


"""
GSS(f)
performs discrete 1D scale-space of signal and tracks minima through 
scale-space
Input:
    f       - input signal
Output:
    plane   - 2D plot of minima paths through scale-space representation of f
Author: Basile Hurat, Jerome Gilles"""
def GSS(f):
    t = 0.5
    n = 3
    num_iter = 1 * np.max([np.ceil(len(f) / n), 3])
    #First, define scale-space kernel (discrete Gaussian kernel)
    ker = np.exp(-t) * iv(np.arange(-n, n + 1), t)
    ker = ker/np.sum(ker)

    #Initialize place to store result of each layer GSS
    plane = np.zeros([len(f), num_iter.astype(int) + 1])
    plane[:, 0] = localmin(f)

    #Iterate through scalespace and store minima at each scale
    for i in range(1, num_iter.astype(int) + 1):
        f = np.pad(f, n, 'reflect')
        f = np.convolve(f, ker, 'same')
        f = f[n:-n]
        plane[:, i] = localmin(f)
        if np.sum(plane[:, i]) <= 2:
            break
    return plane

"""
lengthScaleCurve(plane)
Given the 2D plot of minima paths in scale-space representation, this function
extracts the persistence of each minima, as well as their starting position in 
signal
Input:
    plane   - 2D plot of minima paths through scale-space representation
Output:
    lengths - persistence of each minimum
    indices - position of each minimum
Author: Basile Hurat, Jerome Gilles"""
def lengthScaleCurve(plane):
    [w,num_iter] = plane.shape
    num_curves = np.sum(plane[:, 0])
    lengths = np.ones(num_curves.astype(int))
    indices = np.zeros(num_curves.astype(int))
    current_curve = 0
 
    for i in range(0, w):
        if plane[i, 0] == 1:
            indices[current_curve] = i
            i0 = i
            height = 2
            stop = 0
            while stop == 0:
                flag = 0
                for p in range(-1, 2):
                    if (i + p  < 0) or (i + p >= w):
                        continue
                    #If minimum at next iteration of scale-space, increment length
                    #height, minimum location
                    if plane[i + p, height] == 1:                         
                        lengths[current_curve] += 1 
                        height += 1
                        i0 += p
                        flag = 1
                        #Stop if pas number of iterations
                        if height >= num_iter:  
                            stop = 1
                        break
                #Stop if no minimum found
                if flag == 0:
                    stop = 1
            #Go to next curve/minimum after done
            current_curve += 1  

    return [lengths, indices]


"""
localmin(f):
Givan an array f, returns a boolean array that represents positions of local 
minima - Note, handles minima plateaus by assigning all points on plateau as 
minima
Input:
    f       - an array of numbers
Output:
    minima  - boolean array of same length as f, which represents positions of 
            local minima in f
Author: Basile Hurat, Jerome Gilles"""
def localmin(f):
    w = len(f)
    minima = np.zeros(w)
    for i in range(0, w):
        minima[i] = 1
        right = 1
        while 1:
            if i - right >= 0:
                if f[i - right] < f[i]:
                    minima[i] = 0
                    break
                elif f[i - right] == f[i]:
                    right += 1
                else:
                    break
            else:
                break
        if minima[i] == 1:
            left = 1
            while 1:
                if i + left < w:
                    if f[i + left] < f[i]:
                        minima[i] = 0
                        break
                    elif f[i + left] == f[i]:
                        left += 1
                    else:
                        break
                else:
                    break
    i = 0
    while i < w:
        if minima[i] == 1:
            j = i
            flat_count = 1
            flag = 0
            while (j + 1 < w) and (minima[j + 1] == 1):
                minima[j] = 0
                flat_count += 1
                j += 1
                flag = 1
                
            if flag == 1:
                minima[j - np.floor(flat_count/2).astype(int)] = 1
                minima[j] = 0
                i = j
        i += 1
    minima = removePlateaus(minima)
    minima[0] = 0
    minima[-1] = 0
    return minima

def removePlateaus(x):
    w = len(x); i = 0
    flag = 0
    while i < w:
        if x[i] == 1:
            plateau = 1
            while 1:
                if i + plateau < w and x[i + plateau] == 1:
                    plateau += 1
                    print(f'{i}, plateau = {plateau}')
                else:
                    flag = plateau > 1
                    break
        if flag:
            x[i:i + plateau] = 0
            x[i + plateau // 2] = 1
            i = i + plateau
        else:
            i += 1
    return x
    
"""
otsu(lengths)
2-class classification method which minimizes the inter-class variance of the 
class probability and the respective class averages
Input:
    lengths - array to be thresholded or classified
Output:
    thresh  - detected threshold that separates the two classes
Author: Basile Hurat, Jerome Gilles"""
def otsu(lengths):
    hist_max = np.max(lengths); 
    histogram = np.histogram(lengths, hist_max.astype(int))[0]
    hist_normalized = histogram / np.sum(histogram) #normalize
    Q = hist_normalized.cumsum()
    
    bins = np.arange(hist_max)
    fn_min = np.inf
    thresh = -1

    for i in range(1, hist_max.astype(int)):
        p1, p2 = np.hsplit(hist_normalized, [i])
        q1, q2 = Q[i], Q[hist_max.astype(int) - 1] - Q[i]
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i]) #weights

        #Means and variances
        m1 = np.sum(p1 * b1) / q1
        m2 = np.sum(p2 * b2) / q2
        v1 = np.sum((b1 - m1) ** 2 * p1) / q1
        v2 = np.sum((b2 - m2) ** 2 * p2) / q2

        #calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh

"""
empiricalLaw(lengths)
2-class classification method which classifies by considering lengths which are
epsilon meaningful for an empirical law. 
Input:
    lengths     - array to be thresholded or classified
Output:
    meaningful  - boolean array of where meaningful lengths are
Author: Basile Hurat, Jerome Gilles"""
def empiricalLaw(lengths):
    hist_max = np.max(lengths); 
    histogram = np.histogram(lengths, hist_max.astype(int))[0]
    hist_normalized = histogram / np.sum(histogram) #normalize
    Q = hist_normalized.cumsum()
    meaningful = np.where(Q > (1 - 1 / len(lengths)))[0][0] + 1
    return meaningful
    
"""
halfNormal(lengths)
2-class classification method which classifies by considering lengths which are
epsilon-meaningful for a half-normal law fitted to the data. 
Input:
    lengths     - array to be thresholded or classified
Output:
    thresh  - detected threshold that separates the two classes
Author: Basile Hurat, Jerome Gilles"""
def halfNormal(lengths):
    sigma=np.sqrt(np.pi) * np.mean(lengths)
    thresh = sigma * erfinv(erf(np.max(lengths) / sigma) - 1 / len(lengths))
    return thresh

"""
ewtkmeans(lengths)
1D k-means clustering function for 2 class classification
Input:
    lengths     - array to be clustered or classified
Output:
    closest  - label array that gives final classification 
Author: Basile Hurat, Jerome Gilles"""
def ewtkmeans(lengths,maxIter):
    k = 2
    centroids = np.zeros(k)
    distances = np.inf * np.ones([len(lengths), k])
    closest = -np.ones([len(lengths), 2])
    closest[:, 0] = distances[:, 0]
    breaker = 0
    for i in range(0, k):
        centroids[i] = np.random.uniform(1, np.max(lengths))
    while breaker < maxIter:
        prev_closest = closest[:, 1]
        for i in range(0, k):
            distances[:, i] = np.abs(lengths - centroids[i])
            closest[distances[:, i] < closest[:,0], 1] = i
            closest[distances[:, i] < closest[:,0], 0] = distances[distances[:, i] < closest[:, 0], i]
        if np.all(closest[:, 1] == prev_closest):
            break
        for i in range(0, i):
            centroids[i] = np.mean(lengths[closest[:, 1] == i])
        if breaker == maxIter - 1:
            print('k-means did not converge')
        breaker += 1
    return closest[:, 1]


"""
ewt2dTensor(f,params)
Given an image, function performs the 2D Tensor empirical wavelet transform - 
This is a separable approach that contructs a filter bank based on detected 
rectangular supports
Input:
    f           - 2D array containing signal
    params      - parameters for EWT (see utilities)
Output:
    ewtc        - empirical wavelet coefficients
    mfb_row     - constructed Tensor empirical wavelet filter bank for rows
    mfb_col     - constructed Tensor empirical wavelet filter bank for columns
    bounds_row  - detected row boundaries in [0,pi]
    bounds_col  - detected column boundaries in [0,pi]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dTensor(f,params):
    ff = np.fft.fft(f, axis=0) #take 1D fft along columns
    meanfft = np.sum(np.abs(ff), axis=1) / ff.shape[0] #take average of magnitude
    bounds_col = ewt_boundariesDetect(meanfft, params)
    bounds_col = bounds_col * 2 * np.pi / len(meanfft)
    
    ewtc_col = []
    mfb_col = ewt_LP_Filterbank(bounds_col, len(meanfft), params.real) #construct filter bank
    for i in range(0, len(mfb_col)):
        filter_col = np.tile(mfb_col[i], [f.shape[1], 1]).T #repeat down
        ewtc_col.append(np.real(np.fft.ifft(filter_col * ff, axis=0))) #get coefficients,
    
    # REPEAT WITH ROWS #
    ff = np.fft.fft(f, axis=1) #take 1D fft along rows
    meanfft = np.mean(np.abs(ff.T), axis=1) 
    bounds_row = ewt_boundariesDetect(meanfft, params)
    bounds_row = bounds_row * 2 * np.pi / len(meanfft)
    
    ewtc = []
    mfb_row = ewt_LP_Filterbank(bounds_row, len(meanfft), params.real)
    for i in range(0, len(mfb_row)):
        ewtc_row = []
        filter_row = np.tile(mfb_row[i], [f.shape[0], 1])
        for j in range(0,len(mfb_col)):
            ff = np.fft.fft(ewtc_col[j])
            ewtc_row.append(np.real(np.fft.ifft(filter_row * ff, axis = 1)))
        ewtc.append(ewtc_row)
            
    return [ewtc, mfb_row, mfb_col, bounds_row, bounds_col]

"""
iewt2dTensor(ewtc,mfb_row,mfb_col)
Performs the inverse Tensor 2D empirical wavelet transform given the Tensor EWT
ceofficients and filters
Input:
    ewtc    - Tensor empirical wavelet coefficients
    mfb_row - corresponding Tensor empirical wavelet filter bank for rows
    mfb_col - corresponding Tensor empirical wavelet filter bank for columns
Output:
    img     - reconstructed image
Author: Basile Hurat, Jerome Gilles"""  
def iewt2dTensor(ewtc, mfb_row, mfb_col):
    [h, w]= ewtc[0][0].shape
    ewt_col = []
    for i in range(0, len(mfb_col)):
        for j in range(0, len(mfb_row)):
            ff = np.fft.fft(ewtc[j][i])
            filter_row = np.tile(mfb_row[j], [h, 1])
            if j == 0:
                ewt_col.append(np.zeros([h, w]))
            ewt_col[i] += np.real(np.fft.ifft(filter_row * ff))
    
    for i in range(0, len(mfb_col)):
        ff = np.fft.fft(ewt_col[i].T)
        if i == 0:
            img = np.zeros([h, w])
        
        filter_col = np.tile(mfb_col[i], [w, 1])
        img = img + np.real(np.fft.ifft(filter_col * ff)).T
    return img

"""
ewt2dLP(f,params)
Given an image, function performs the 2D Littlewood-Paley empirical wavelet 
transform - This is an approach that contructs a filter bank of anulli based on 
detected scales. 
Input:
    f               - vector x as an input
    params          - parameters for EWT (see utilities)
Output:
    ewtc            - empirical wavelet coefficients
    mfb             - constructed Curvelet empirical wavelet filter bank
    bounds_scales   - detected scale boundaries in [0,pi]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dLP(f,params):
    [h, w] = f.shape
    #get boundaries
    ppff = ppfft(f)
    meanppff = np.fft.fftshift(np.mean(np.abs(ppff), axis=1))
    bounds_scales = ewt_boundariesDetect(meanppff, params)
    bounds_scales *= np.pi / np.ceil((len(meanppff) / 2))
   
    #construct filter bank
    mfb = ewt2d_LPFilterbank(bounds_scales, h, w)
    
    #filter out coefficients
    ff = np.fft.fft2(f)
    ewtc = []
    for i in range(0, len(mfb)):
        ewtc.append(np.real(np.fft.ifft2(mfb[i] * ff)))
    return [ewtc, mfb, bounds_scales]

"""
iewt2dLP(ewtc,mfb)
Performs the inverse Littlewood-Paley 2D empirical wavelet transform given the 
Littlewood-Paley EWT ceofficients and filters
Input:
    ewtc    - Littlewood-Paley empirical wavelet coefficients
    mfb     - corresponding Littlewood-Paley filter bank
Output:
    recon   - reconstructed image
Author: Basile Hurat, Jerome Gilles"""  
def iewt2dLP(ewtc, mfb):
    recon = np.fft.fft2(ewtc[0]) * mfb[0]
    for i in range(1, len(mfb)):
        recon += np.fft.fft2(ewtc[i]) * mfb[i]
    recon = np.real(np.fft.ifft2(recon))
    return recon
        
"""
ewt2d_LPFilterbank(bounds_scales,h,w)
Constructs the Littlewood-Paley 2D EWT filter bank with filters of size [h,w]
based on a set of detected scales
Input:
    bounds_scales   - detected scale bounds in range [0,pi]
    h               - desired height of filters
    w               - desired width of filters
Output:
    mfb             - Littlewood-Paley 2D EWT filter bank
Author: Basile Hurat, Jerome Gilles"""  
def ewt2d_LPFilterbank(bounds_scales, h, w):
    if h % 2 == 0:
        h += 1
        h_extended = 1
    else:
        h_extended = 0
    if w % 2 == 0:
        w += 1
        w_extended = 1
    else:
        w_extended = 0
    #First, we calculate gamma for scales
    gamma_scales = np.pi
    for k in range(0, len(bounds_scales) - 1):
        r = (bounds_scales[k + 1] - bounds_scales[k]) / (bounds_scales[k + 1] + bounds_scales[k])
        if r < gamma_scales and r > 1e-16:
            gamma_scales = r
    r = (np.pi - bounds_scales[-1]) / (np.pi + bounds_scales[-1]) #check last bound
    if r < gamma_scales and r > 1e-16:
        gamma_scales = r

    if gamma_scales > bounds_scales[0]:     #check first bound
        gamma_scales = bounds_scales[0]
    gamma_scales *= (1 - 1 / max(h, w)) #guarantees that we have strict inequality
    radii = np.zeros([h, w])
    
    h_center = h // 2 + 1
    w_center = w // 2 + 1
    for i in range(0, h):
        for j in range(0, w):
            ri = (i + 1.0 - h_center) * np.pi / h_center
            rj = (j + 1.0 - w_center) * np.pi / w_center
            radii[i, j] = np.sqrt(ri ** 2 + rj ** 2)
    
    mfb = []
    mfb.append(ewt2d_LPscaling(radii, bounds_scales[0], gamma_scales))
    for i in range(0, len(bounds_scales) - 1):
        mfb.append(ewt2d_LPwavelet(radii, bounds_scales[i], bounds_scales[i + 1], gamma_scales))
    mfb.append(ewt2d_LPwavelet(radii, bounds_scales[-1], 2 * np.pi, gamma_scales))
    
    if h_extended == 1: #if we extended the height of the image, trim
        h -= 1
        for i in range(0, len(mfb)):
            mfb[i] = mfb[i][0:-1, :]
    if w_extended == 1: #if we extended the width of the image, trim
        w -= 1
        for i in range(0, len(mfb)):
            mfb[i] = mfb[i][:, 0:-1]
    #invert the fftshift since filters are centered
    for i in range(0, len(mfb)):
        mfb[i] = np.fft.ifftshift(mfb[i])
            
    #Resymmetrize for even images
    if h_extended == 1:
        s = np.zeros(w)
        if w % 2 == 0:
            mfb[-1][h // 2, 1:w // 2] += mfb[-1][h // 2, -1:w // 2:-1]
            mfb[-1][h // 2, w // 2 + 1:] = mfb[-1][h // 2, w // 2 - 1:0:-1]
            s += mfb[-1][h // 2, :] ** 2
            #normalize for tight frame
            mfb[-1][h // 2, 1:w // 2] /= np.sqrt(s[1:w // 2])
            mfb[-1][h // 2, w // 2 + 1:] /= np.sqrt(s[w // 2 + 1:])
        else:
            mfb[-1][h // 2, 0:w // 2] += mfb[-1][h // 2, -1:w // 2:-1]
            mfb[-1][h // 2, w // 2 + 1:] = mfb[-1][h // 2, w // 2-1::-1]
            s += mfb[-1][h // 2, :] ** 2
            #normalize for tight frame
            mfb[-1][h // 2, 0:w // 2]  /= np.sqrt(s[0:w // 2])
            mfb[-1][h // 2, w // 2 + 1:] /= np.sqrt(s[w // 2 + 1:])
    if w_extended == 1:
        s = np.zeros(h)
        if h%2 == 0:
            mfb[-1][1:h // 2, w // 2] += mfb[-1][-1:h // 2:-1, w // 2]
            mfb[-1][h // 2 + 1:, w // 2] = mfb[-1][h // 2 - 1:0:-1, w // 2]
            s += mfb[-1][:, w // 2] ** 2
            #normalize for tight frame
            mfb[-1][1:h // 2, w // 2] /= np.sqrt(s[1:h // 2])
            mfb[-1][h // 2 + 1:, w // 2] /= np.sqrt(s[h // 2 + 1:]) 
        else:
            mfb[-1][0 : h // 2, w // 2] += mfb[-1][-1:h // 2:-1, w // 2]
            mfb[-1][h // 2 + 1:, w // 2] = mfb[-1][h // 2 - 1::-1, w // 2]
            s += mfb[-1][:, w // 2] ** 2
            #normalize for tight frame
            mfb[-1][0:h // 2, w // 2] /= s[0:h // 2]
            mfb[-1][h // 2 + 1:, w // 2] /= s[h // 2 + 1:]
    return mfb

"""
ewt2d_LPscaling(radii,bound0,gamma)
Constructs the empirical Littlewood-Paley scaling function (circle)
Input:
    radii   - reference image where pixels are equal to their distance from 
            center
    bound0   - first radial bound
    gamma   - detected scale gamma to guarantee tight frame
Output:
    scaling - resulting empirical Littlewood-Paley scaling function
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_LPscaling(radii, bound0, gamma):
    an = 1 / (2 * gamma * bound0) 
    mbn = (1 - gamma) * bound0 # inner circle up to beginning of transtion
    pbn = (1 + gamma) * bound0 #end of transition
    
    scaling = 0 * radii #initiate w/ zeros
    scaling[radii < mbn] = 1
    scaling[radii >= mbn] = np.cos(np.pi * ewt_beta(an * (radii[radii >= mbn] - mbn)) / 2)
    scaling[radii > pbn] = 0
    return scaling

"""
ewt2d_LPwavelet(radii,bound1,bound2,gamma)
Constructs the empirical Littlewood-Paley wavelet function (annulus)
Input:
    radii   - reference image where pixels are equal to their distance from 
            center
    bound1  - lower radial bound
    bound2  - upper radial bound
    gamma   - detected gamma to guarantee tight frame
Output:
    wavelet - resulting empirical Littlewood-Paley wavelet
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_LPwavelet(radii, bound1, bound2, gamma):
    wan = 1 / (2 * gamma * bound1) #scaling factor
    wam = 1 / (2 * gamma * bound2) 
    wmbn = (1 - gamma) * bound1 #beginning of lower transition
    wpbn = (1 + gamma) * bound1 #end of lower transition
    wmbm = (1 - gamma) * bound2  #beginning of upper transition
    wpbm = (1 + gamma) * bound2 #end of upper transition
    
    wavelet = 0 * radii #initialize w/ zeros
    inside = (radii > wmbn) * (radii < wpbm)
    wavelet[inside] = 1.0 #set entire angular wedge equal to 1
    temp = inside*(radii >= wmbm) * (radii <= wpbm) #upper transition
    wavelet[temp] *= np.cos(np.pi * ewt_beta(wam * (radii[temp] - wmbm)) / 2)
    temp = inside * (radii >= wmbn) * (radii <= wpbn) #lower transition
    wavelet[temp] *= np.sin(np.pi * ewt_beta(wan * (radii[temp] - wmbn)) / 2)
    return wavelet

"""
ewt2dRidgelet(f,params)
Given an image, function performs the 2D Littlewood-Paley empirical wavelet 
transform - This is an approach that contructs a filter bank of anulli based on 
detected scales. 
Input:
    f               - vector x as an input
    params          - parameters for EWT (see utilities)
Output:
    ewtc            - empirical wavelet coefficients
    mfb             - constructed Curvelet empirical wavelet filter bank
    bounds_scales   - detected scale boundaries in [0,pi]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dRidgelet(f, params):
    [h, w] = f.shape
    #get boundaries
    ppff = ppfft(f)
    meanppff = np.fft.fftshift(np.mean(np.abs(ppff), axis=1))
    bounds_scales = ewt_boundariesDetect(meanppff, params)
    bounds_scales *= np.pi / np.ceil((len(meanppff) / 2))
    
    #Construct 1D filterbank
    mfb_1d = ewt_LP_Filterbank(bounds_scales, ppff.shape[0], params.real)
    
    #filter out coefficients
    mfb = []
    ewtc = []
    for i in range(0, len(mfb_1d)):
        mfb.append(np.tile(mfb_1d[i], [ppff.shape[1],1]).T)
        ewtc.append(np.real(np.fft.ifft(mfb[i] * np.fft.fftshift(ppff, 0),axis=0)))
    return [ewtc, mfb, bounds_scales]

"""
iewt2dRidgelet(ewtc,mfb)
Performs the inverse Curvelet 2D empirical wavelet transform given the Curvelet
EWT ceofficients and filters
Input:
    ewtc    - Curvelet empirical wavelet coefficients
    mfb     - corresponding Curvelet filter bank
Output:
    recon   - reconstructed image
Author: Basile Hurat, Jerome Gilles""" 

def iewt2dRidgelet(ewtc, mfb, isOdd=0):
   ppff=1j * np.zeros(ewtc[0].shape)
   for i in range(0, len(mfb)):
       ppff += np.fft.fftshift(np.fft.fft(ewtc[i], axis=0) * mfb[i], 0)
   recon = np.real(ippfft(ppff, 1e-10))
   if isOdd:
       recon = recon[0:-1, 0:-1]
   return recon

"""
ewt2dCurvelet(f,params)
Given an image, function performs the 2D Curvelet empirical wavelet transform - 
This is an approach that contructs a filter bank of polar wedges based on 
detected scales and orientations. There are three options
    Option 1 - Detects scales and orientations separately
    Option 2 - Detects scales first, and then orientations for each scale
    Option 3 - Detects orientations first, and then scales for each orientation
Input:
    f               - vector x as an input
    params          - parameters for EWT (see utilities)
Output:
    ewtc            - empirical wavelet coefficients
    mfb             - constructed Curvelet empirical wavelet filter bank
    bounds_scales  - detected scale boundaries in [0,pi]
    bounds_angles  - detected angle boundaries in [-3pi/4,pi/4]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dCurvelet(f, params):
    [h, w] = f.shape
    ppff = ppfft(f)
    
    if params.option == 1:
    #Option 1: Computes scales and angles independently    
        #Begin with scales
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff), axis=1))
        bounds_scales = ewt_boundariesDetect(meanppff, params)
        bounds_scales *= np.pi / np.ceil((len(meanppff) / 2))
        
        #Then do with angles
        meanppff = np.mean(np.abs(ppff), axis=0)
        bounds_angles = ewt_boundariesDetect(meanppff, params)
        bounds_angles = bounds_angles * np.pi / np.ceil((len(meanppff) / 2)) - np.pi * .75
    
    elif params.option == 2:
        #Option 2: Computes Scales first, then angles 
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff), axis=1))
        bounds_scales = ewt_boundariesDetect(meanppff, params)
        bounds_angles = []
        for i in range(0, len(bounds_scales) - 1):
            meanppff = np.mean(np.abs(ppff[int(bounds_scales[i]):int(bounds_scales[i + 1] + 1), :]), axis=0)
            bounds =  ewt_boundariesDetect(meanppff, params)
            #append
            bounds_angles.append(bounds * np.pi / np.ceil((len(meanppff) / 2)) - np.pi * .75)
            
        #Do last linterval
        meanppff = np.mean(np.abs(ppff[int(bounds_scales[-1]):, :]), axis=0)
        bounds =  ewt_boundariesDetect(meanppff, params)
        #append
        bounds_angles.append(bounds * np.pi / np.ceil((len(meanppff) / 2)) - np.pi * .75)
        #normalize scale bounds
        bounds_scales *= np.pi / np.ceil((len(meanppff) / 2))
        
    elif params.option == 3:
        #Option 3: Computes angles, then scales
        bounds_scales = []
        #Get first scale
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff), axis=1))
        LL = len(meanppff) // 2
        #bound0 = ewt_boundariesDetect(meanppff[LL:],params)[0]
        bound0 = ewt_boundariesDetect(meanppff, params)[0]
        bounds_scales.append([bound0 * np.pi / LL])
        bound0 = int(bound0)
        #Compute mean-pseudo-polar fft for angles, excluding first scale to 
        #find angle bounds
        #meanppff = np.mean(np.abs(ppff[ppff.shape[0]//2+bound0:,:]),0)
        meanppff = np.mean(np.abs(ppff[bound0:-bound0, :]), axis=0)
        bounds_theta = ewt_boundariesDetect(meanppff, params, sym=0)
        bounds_angles = (bounds_theta - 1) * np.pi / len(meanppff) - 0.75 * np.pi
        bounds_theta = bounds_theta.astype(int)
        #Now we find scale bounds at each angle
        for i in range(0, len(bounds_theta) - 1):
            #meanppff = np.mean(np.abs(ppff[LL+bound0:,bounds_theta[i]:bounds_theta[i+1]+1]),1)
            meanppff = np.fft.fftshift(
                np.mean(
                    np.abs(
                        ppff[bound0:-bound0, bounds_theta[i]:bounds_theta[i + 1] + 1]), 1))
            bounds = ewt_boundariesDetect(meanppff, params)
            bounds_scales.append((bounds + bound0) * np.pi / LL)
        
        #and also for the last angle
        #meanppff = np.mean(np.abs(ppff[LL+bound0:,bounds_theta[-1]:]),1)
        #meanppff += np.mean(np.abs(ppff[LL+bound0:,1:bounds_theta[0]+1]),1)
        meanppff = np.mean(np.abs(ppff[bound0:-bound0, bounds_theta[-1]:]), 1)
        meanppff += np.mean(np.abs(ppff[bound0:-bound0, 1:bounds_theta[0] + 1]), 1)
        
        params.spectrumRegularize = 'closing'
        bounds = ewt_boundariesDetect(np.fft.fftshift(meanppff), params)
        bounds_scales.append((bounds + bound0) * np.pi / LL)
    else:
        print('invalid option')
        return -1
    #Once bounds are found, construct filter bank, take fourier transform of 
    #image, and filter
    mfb = ewt2d_curveletFilterbank(bounds_scales, bounds_angles, h, w, params.option)
    ff = np.fft.fft2(f)
    ###ewtc = result!
    ewtc = []
    for i in range(0, len(mfb)):
        ewtc_scales = []
        for j in range(0, len(mfb[i])):
            ewtc_scales.append(np.real(np.fft.ifft2(mfb[i][j] * ff)))
        ewtc.append(ewtc_scales)
    return [ewtc, mfb, bounds_scales, bounds_angles]

"""
iewt2dCurvelet(ewtc,mfb)
Performs the inverse Curvelet 2D empirical wavelet transform given the Curvelet
EWT ceofficients and filters
Input:
    ewtc    - Curvelet empirical wavelet coefficients
    mfb     - corresponding Curvelet filter bank
Output:
    recon   - reconstructed image
Author: Basile Hurat, Jerome Gilles"""  
def iewt2dCurvelet(ewtc, mfb):
    recon = np.fft.fft2(ewtc[0][0]) * mfb[0][0]
    for i in range(1, len(mfb)):
        for j in range(0, len(mfb[i])):
            recon += np.fft.fft2(ewtc[i][j]) * mfb[i][j]
    recon = np.real(np.fft.ifft2(recon))
    return recon

"""
ewt_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
Constructs the Curvelet 2D empirical wavelet filter bank 
Input:
    bounds_scales   - detected scale boundaries
    bounds_angles   - detected orientation boundaries
    h               - desired height of filters
    w               - desired width of filters
    option          - Option for curvelet
                        1 for separate detection of scales and orientations
                        2 for detection of scales and then orientations 
                        3 for detection of orientations and then scales
Output:
    mfb             - resulting filter bank 
Author: Basile Hurat, Jerome Gilles"""      
def ewt2d_curveletFilterbank(bounds_scales, bounds_angles, h, w, option):
    if h % 2 == 0:
        h += 1
        h_extended = 1
    else:
        h_extended = 0
    if w % 2 == 0:
        w += 1
        w_extended = 1
    else:
        w_extended = 0
    
    if option == 1:
        #Scales and angles detected separately
        
        #First, we calculate gamma for scales
        gamma_scales = np.pi
        for k in range(0, len(bounds_scales) - 1):
            r = (bounds_scales[k + 1] - bounds_scales[k]) / (bounds_scales[k + 1] + bounds_scales[k])
            if r < gamma_scales and r > 1e-16:
                gamma_scales = r
        
        r = (np.pi - bounds_scales[-1]) / (np.pi + bounds_scales[-1]) #check last bound
        if r < gamma_scales and r > 1e-16:
            gamma_scales = r
        if gamma_scales > bounds_scales[0]:     #check first bound
            gamma_scales = bounds_scales[0]
        gamma_scales *= (1 - 1 / max(h, w)) #guarantees that we have strict inequality
        
        #Get gamma for angles
        gamma_angles = 2 * np.pi
        for k in range(0, len(bounds_angles) - 1):
            r = (bounds_angles[k + 1] - bounds_angles[k]) / 2
            if r < gamma_angles and r > 1e-16:
                gamma_angles = r
        r = (bounds_angles[0] + np.pi - bounds_angles[-1]) / 2 #check extreme bounds (periodic)
        if r < gamma_angles and r > 1e-16:
            gamma_angles = r
        gamma_angles *= (1 - 1 / max(h, w)) #guarantees that we have strict inequality    
        
        #construct matrices representing radius and angle value of each pixel
        radii = np.zeros([h, w])
        theta = np.zeros([h, w])
        h_center = h // 2 + 1; w_center = w // 2 + 1
        for i in range(0, h):
            for j in range(0, w):
                ri = (i + 1.0 - h_center) * np.pi / h_center
                rj = (j + 1.0 - w_center) * np.pi / w_center
                radii[i, j] = np.sqrt(ri ** 2 + rj ** 2)
                theta[i, j] = np.arctan2(ri, rj)
                if theta[i, j] < -.75 * np.pi:
                    theta[i, j] += 2 * np.pi
        
        mfb = []
        #construct scaling
        mfb.append([ewt2d_curveletScaling(radii, bounds_scales[0], gamma_scales)])
        
        #construct angular wedges for all but last scales
        for i in range(0, len(bounds_scales) - 1):
            mfb_scale = []           
            for j in range(0, len(bounds_angles) - 1):
                mfb_scale.append(
                    ewt2d_curveletWavelet(
                        theta,
                        radii,
                        bounds_angles[j],
                        bounds_angles[j + 1],
                        bounds_scales[i],
                        bounds_scales[i + 1],
                        gamma_angles,
                        gamma_scales
                        )
                        )
            mfb_scale.append(
                ewt2d_curveletWavelet(
                    theta,
                    radii,
                          bounds_angles[-1],
                          bounds_angles[0] + np.pi,
                          bounds_scales[i], 
                          bounds_scales[i + 1],
                          gamma_angles,
                          gamma_scales
                          )
                          )
            mfb.append(mfb_scale)
        #construct angular wedges for last scales
    
        mfb_scale = []            
        for j in range(0, len(bounds_angles) - 1):
            mfb_scale.append(
                ewt2d_curveletWavelet(
                    theta,
                    radii,
                    bounds_angles[j],
                    bounds_angles[j + 1],
                    bounds_scales[-1], 
                    np.pi * 2,
                    gamma_angles,
                    gamma_scales
                    )
                    )
        mfb_scale.append(
            ewt2d_curveletWavelet(
                theta,
                radii,
                bounds_angles[-1],
                bounds_angles[0] + np.pi,
                bounds_scales[-1],
                np.pi * 2,
                gamma_angles,
                gamma_scales
                )
                )

        mfb.append(mfb_scale)
    elif option == 2:
        #Scales detected first, and then angles
        #Get gamma for scales
        gamma_scales = np.pi
        for k in range(0, len(bounds_scales) - 1):
            r = (bounds_scales[k + 1] - bounds_scales[k]) / (bounds_scales[k + 1] + bounds_scales[k])
            if r < gamma_scales and r > 1e-16:
                gamma_scales = r
        
        r = (np.pi - bounds_scales[-1]) / (np.pi + bounds_scales[-1]) #check last bound
        if r < gamma_scales and r > 1e-16:
            gamma_scales = r
        if gamma_scales > bounds_scales[0]:     #check first bound
            gamma_scales = bounds_scales[0]
        gamma_scales *= (1 - 1 / max(h, w)) #guarantees that we have strict inequality
        
        #Get gammas for angles
        gamma_angles = 2 * np.pi * np.ones(len(bounds_scales))
        for i in range(0, len(gamma_angles)):
            for k in range(0, len(bounds_angles[i]) - 1):
                r = (bounds_angles[i][k + 1] - bounds_angles[i][k]) / 2
                if r < gamma_angles[i] and r > 1e-16:
                    gamma_angles[i] = r
            r = (bounds_angles[i][0] + np.pi - bounds_angles[i][-1]) / 2 #check extreme bounds (periodic)
            if r < gamma_angles[i] and r > 1e-16:
                gamma_angles[i] = r
        gamma_angles *= (1 - 1 / max(h, w)) #guarantees that we have strict inequality    
        
        #construct matrices representing radius and angle value of each pixel
        radii = np.zeros([h, w])
        theta = np.zeros([h, w])
        h_center = h // 2 + 1; w_center = w // 2 + 1
        for i in range(0, h):
            for j in range(0, w):
                ri = (i + 1.0 - h_center) * np.pi / h_center
                rj = (j + 1.0 - w_center) * np.pi / w_center
                radii[i, j] = np.sqrt(ri ** 2 + rj ** 2)
                theta[i, j] = np.arctan2(ri, rj)
                if theta[i, j] < -.75 * np.pi:
                    theta[i, j] += 2 * np.pi
        
        mfb = []
        #construct scaling
        mfb.append([ewt2d_curveletScaling(radii, bounds_scales[0], gamma_scales)])
        
        #construct angular wedges for all but last scales
        for i in range(0, len(bounds_scales) - 1):
            mfb_scale = []           
            for j in range(0, len(bounds_angles[i]) - 1):
                mfb_scale.append(
                    ewt2d_curveletWavelet(
                        theta,
                        radii,
                        bounds_angles[i][j], 
                        bounds_angles[i][j + 1],
                        bounds_scales[i],
                        bounds_scales[i + 1],
                        gamma_angles[i],
                        gamma_scales
                        )
                        )
            mfb_scale.append(
                ewt2d_curveletWavelet(
                    theta,
                    radii,
                    bounds_angles[i][-1],
                    bounds_angles[i][0] + np.pi,
                    bounds_scales[i],
                    bounds_scales[i + 1],
                    gamma_angles[i],
                    gamma_scales
                    )
                    )
            mfb.append(mfb_scale)
        #construct angular wedges for last scales
    
        mfb_scale = []            
        for j in range(0, len(bounds_angles[-1]) - 1):
            mfb_scale.append(
                ewt2d_curveletWavelet(
                    theta,
                    radii,
                    bounds_angles[-1][j],
                    bounds_angles[-1][j + 1],
                    bounds_scales[-1],
                    np.pi * 2,
                    gamma_angles[-1],
                    gamma_scales
                    )
                    )
        mfb_scale.append(
            ewt2d_curveletWavelet(
                theta,
                radii,
                bounds_angles[-1][-1],
                bounds_angles[-1][0] + np.pi,
                bounds_scales[-1],
                np.pi * 2,
                gamma_angles[-1],
                gamma_scales
                )
                )
        mfb.append(mfb_scale)
    elif option == 3:
        # Angles detected first then scales per angles
        #compute gamma for theta
        gamma_angles = 2 * np.pi
        for k in range(0, len(bounds_angles) - 1):
            r = (bounds_angles[k + 1] - bounds_angles[k]) / 2
            if r < gamma_angles and r > 1e-16:
                gamma_angles = r
        r = (bounds_angles[0] + np.pi - bounds_angles[-1]) / 2 #check extreme bounds (periodic)
        if r < gamma_angles and r > 1e-16:
            gamma_angles = r
        gamma_angles *= (1 - 1 / max(h, w)) #guarantees that we have strict inequality    
        
        #compute gamma for scales
        gamma_scales = bounds_scales[0][0] / 2
        for i in range(1, len(bounds_angles)):
            for j in range(0, len(bounds_scales[i]) - 1):
                r = (bounds_scales[i][j + 1] - bounds_scales[i][j])\
                    /(bounds_scales[i][j + 1] + bounds_scales[i][j])
                if r < gamma_scales and r > 1e-16:
                    gamma_scales = r
            r = (np.pi - bounds_scales[i][-1]) / (np.pi + bounds_scales[i][-1])
            if r < gamma_scales and r > 1e-16:
                gamma_scales = r
        gamma_scales *= (1 - 1 / max(h, w))
        
        radii = np.zeros([h, w])
        theta = np.zeros([h, w])
        h_center = h // 2 + 1
        w_center = w // 2 + 1
        for i in range(0, h):
            for j in range(0, w):
                ri = (i + 1.0 - h_center) * np.pi / h_center
                rj = (j + 1.0 - w_center) * np.pi / w_center
                radii[i, j] = np.sqrt(ri ** 2 + rj ** 2)
                theta[i, j] = np.arctan2(ri, rj)
        
        #Get empirical scaling function
        mfb = []
        mfb.append([ewt2d_curveletScaling(radii, bounds_scales[0][0], gamma_scales)])
        
        #for each angular sector, get empirical wavelet 
        for i in range(0, len(bounds_angles) - 1):
            mfb_scale = []
            #generate first scale
            mfb_scale.append(
                ewt2d_curveletWavelet(
                    theta,
                    radii,
                    bounds_angles[i],
                    bounds_angles[i + 1],
                    bounds_scales[0][0],
                    bounds_scales[i + 1][0],
                    gamma_angles, 
                    gamma_scales
                    )
                    )
            #generate for other scales
            for j in range(0, len(bounds_scales[i + 1]) - 1):
                mfb_scale.append(
                    ewt2d_curveletWavelet(
                        theta,
                        radii,
                        bounds_angles[i],
                        bounds_angles[i + 1],
                        bounds_scales[i + 1][j],
                        bounds_scales[i + 1][j + 1],
                        gamma_angles,
                        gamma_scales
                        )
                        )
            #generate for last scale
            mfb_scale.append(
                ewt2d_curveletWavelet(
                    theta,
                    radii,
                    bounds_angles[i],
                    bounds_angles[i + 1],
                    bounds_scales[i + 1][-1],
                    2 * np.pi,
                    gamma_angles,
                    gamma_scales
                    )
                    )
            mfb.append(mfb_scale)
        
        #Generate for last angular sector
        mfb_scale = []
        mfb_scale.append(
            ewt2d_curveletWavelet(
                theta,
                radii,
                bounds_angles[-1],
                bounds_angles[0] + np.pi,
                bounds_scales[0][0],
                bounds_scales[-1][0],
                gamma_angles,
                gamma_scales
                )
                )
        for i in range(0, len(bounds_scales[-1]) - 1):
            mfb_scale.append(
                ewt2d_curveletWavelet(
                    theta,
                    radii,
                    bounds_angles[-1],
                    bounds_angles[0] + np.pi,
                    bounds_scales[-1][i],
                    bounds_scales[-1][i + 1],
                    gamma_angles,
                    gamma_scales
                    )
                    )
        mfb_scale.append(
            ewt2d_curveletWavelet(
                theta,
                radii,
                bounds_angles[-1],
                bounds_angles[0] + np.pi,
                bounds_scales[-1][-1],
                2 * np.pi,
                gamma_angles,
                gamma_scales
                )
                )
        mfb.append(mfb_scale)
    else:
        print('invalid option')
        return -1
    
    if h_extended == 1: #if we extended the height of the image, trim
        h -= 1
        for i in range(0, len(mfb)):
            for j in range(0, len(mfb[i])):
                mfb[i][j] = mfb[i][j][0:-1, :]
    if w_extended == 1: #if we extended the width of the image, trim
        w -= 1
        for i in range(0, len(mfb)):
            for j in range(0, len(mfb[i])):
                mfb[i][j] = mfb[i][j][:, 0:-1]
    #invert the fftshift since filters are centered
    for i in range(0, len(mfb)):
        for j in range(0, len(mfb[i])):
            mfb[i][j] = np.fft.ifftshift(mfb[i][j])
    #resymmetrize for even images
    if option < 3:
        if h_extended == 1:
            s = np.zeros(w)
            if w%2 == 0:
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][h // 2, 1:w // 2] += mfb[-1][j][h // 2, -1:w // 2:-1]
                    mfb[-1][j][h // 2, w // 2 + 1:] = mfb[-1][j][h // 2, w // 2 - 1:0:-1]
                    s += mfb[-1][j][h // 2,:] ** 2
                #normalize for tight frame
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][h // 2, 1:w // 2] /= np.sqrt(s[1:w // 2])
                    mfb[-1][j][h // 2, w // 2 + 1:] /= np.sqrt(s[w // 2 + 1:])
            else:
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][h // 2, 0:w // 2] += mfb[-1][j][h // 2, -1:w // 2:-1]
                    mfb[-1][j][h // 2, w // 2 + 1:] = mfb[-1][j][h // 2, w // 2 - 1::-1]
                    s += mfb[-1][j][h // 2, :] ** 2
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][h // 2, 0:w // 2]  /= np.sqrt(s[0:w // 2])
                    mfb[-1][j][h // 2, w // 2 + 1:] /= np.sqrt(s[w // 2 + 1:])
        if w_extended == 1:
            s = np.zeros(h)
            if h % 2 == 0:
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][1:h // 2, w // 2] += mfb[-1][j][-1:h // 2:-1, w // 2]
                    mfb[-1][j][h // 2 + 1:, w // 2] = mfb[-1][j][h // 2 - 1:0:-1, w // 2]
                    s += mfb[-1][j][:, w // 2] ** 2
                #normalize for tight frame
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][1:h // 2, w // 2] /= np.sqrt(s[1:h // 2])
                    mfb[-1][j][h // 2 + 1:, w // 2] /= np.sqrt(s[h // 2 + 1:]) 
            else:
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][0:h // 2, w // 2] += mfb[-1][j][-1:h // 2:-1, w // 2]
                    mfb[-1][j][h // 2 + 1:, w // 2] = mfb[-1][j][h // 2 - 1::-1, w // 2]
                    s += mfb[-1][j][:, w // 2] ** 2
                for j in range(0, len(mfb[-1])):
                    mfb[-1][j][0:h // 2, w // 2] /= s[0:h // 2]
                    mfb[-1][j][h // 2 + 1:, w // 2] /= s[h // 2 + 1:]
    else:
        if h_extended == 1:
            s = np.zeros(w)
            if w % 2 == 0:
                for j in range(0, len(mfb)):
                    mfb[j][-1][h // 2, 1:w // 2] += mfb[j][-1][h // 2, -1:w // 2:-1]
                    mfb[j][-1][h // 2, w // 2 + 1:] = mfb[j][-1][h // 2, w // 2 - 1:0:-1]
                    s += mfb[j][-1][h // 2, :] ** 2
                #normalize for tight frame
                for j in range(0,len(mfb)):
                    mfb[j][-1][h // 2, 1:w // 2] /= np.sqrt(s[1:w // 2])
                    mfb[j][-1][h // 2, w // 2 + 1:] /= np.sqrt(s[w // 2 + 1:])
            else:
                for j in range(0, len(mfb)):
                    mfb[j][-1][h // 2, 0:w // 2] += mfb[j][-1][h // 2, -1:w // 2:-1]
                    mfb[j][-1][h // 2, w // 2 + 1:] = mfb[j][-1][h // 2, w // 2 - 1::-1]
                    s += mfb[j][-1][h // 2, :] ** 2
                for j in range(0,len(mfb)):
                    mfb[j][-1][h // 2, 0:w // 2]  /= np.sqrt(s[0:w // 2])
                    mfb[j][-1][h // 2, w // 2 + 1:] /= np.sqrt(s[w // 2 + 1:])
        if w_extended == 1:
            s = np.zeros(h)
            if h % 2 == 0:
                for j in range(0, len(mfb)):
                    mfb[j][-1][1:h // 2, w // 2] += mfb[j][-1][-1:h // 2:-1, w // 2]
                    mfb[j][-1][h // 2 + 1:, w // 2] = mfb[j][-1][h // 2 - 1:0:-1, w // 2]
                    s += mfb[j][-1][:, w // 2] ** 2
                #normalize for tight frame
                for j in range(0, len(mfb)):
                    mfb[j][-1][1:h // 2, w // 2] /= np.sqrt(s[1:h // 2])
                    mfb[j][-1][h // 2 + 1:, w // 2] /= np.sqrt(s[h // 2 + 1:]) 
            else:
                for j in range(0, len(mfb)):
                    mfb[j][-1][0:h // 2, w // 2] += mfb[j][-1][-1:h // 2:-1, w // 2]
                    mfb[j][-1][h // 2 + 1:, w // 2] = mfb[j][-1][h // 2 - 1::-1, w // 2]
                    s += mfb[j][-1][:, w // 2] ** 2
                for j in range(0, len(mfb)):
                    mfb[j][-1][0:h //2, w // 2] /= s[0:h // 2]
                    mfb[j][-1][h // 2 + 1:, w // 2] /= s[h // 2 + 1:]
    return mfb

"""
ewt2d_curveletScaling(radii,bound,gamma)
Constructs the empirical Curvelet scaling function (circle)
Input:
    radii   - reference image where pixels are equal to their distance from 
            center
    bound   - first radial bound
    gamma   - detected scale gamma to guarantee tight frame
Output:
    scaling - resulting empirical Curvelet scaling function
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_curveletScaling(radii, bound, gamma):
    an = 1 / (2 * gamma * bound) 
    mbn = (1 - gamma) * bound # inner circle up to beginning of transtion
    pbn = (1 + gamma) * bound #end of transition
    scaling = 0 * radii #initiate w/ zeros
    scaling[radii < mbn] = 1
    scaling[radii >= mbn] = np.cos(np.pi * ewt_beta(an * (radii[radii>=mbn] - mbn)) / 2)
    scaling[radii > pbn] = 0
    return scaling

"""
ewt2d_curveletWavelet(theta, radii,ang_bound1,ang_bound2, scale_bound1, 
                      scale_bound2,gamma_angle,gamma_scale)
Constructs the empirical Curvelet wavelet function (polar wedge)
Input:
    theta       - reference image where pixels are equal to their angle from center
    radii       - reference image where pixels are equal to their distance from 
                center
    ang_bound1  - lower angular bound
    ang_bound2  - upper angular bound
    scale_bound1- lower radial bound
    scale_bound2- upper radial bound
    gamma_angle - detected angle gamma to guarantee tight frame
    gamma_scale - detected scale gamma to guarantee tight frame
Output:
    wavelet - resulting empirical Curvelet wavelet
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_curveletWavelet(theta, radii, ang_bound1, ang_bound2, scale_bound1, scale_bound2, gamma_angle, gamma_scale):
    #radial parameters
    wan = 1 / (2 * gamma_scale * scale_bound1) #scaling factor
    wam = 1 / (2 * gamma_scale * scale_bound2) 
    wmbn = (1 - gamma_scale) * scale_bound1 #beginning of lower transition
    wpbn = (1 + gamma_scale) * scale_bound1 #end of lower transition
    wmbm = (1 - gamma_scale) * scale_bound2  #beginning of upper transition
    wpbm = (1 + gamma_scale) * scale_bound2 #end of upper transition
    
    #angular parameters 
    an = 1 / (2 * gamma_angle)
    mbn = ang_bound1 - gamma_angle
    pbn = ang_bound1 + gamma_angle
    mbm = ang_bound2 - gamma_angle
    pbm = ang_bound2 + gamma_angle
    
    wavelet = 0 * theta #initialize w/ zeros
    if ang_bound2 - ang_bound1 != np.pi:
        inside = (theta >= mbn) * (theta < pbm) #
    else:
        inside = (theta >= ang_bound1) * (theta <= ang_bound2)
    inside *= (radii > wmbn) * (radii < wpbm)
    wavelet[inside] = 1.0 #set entire angular wedge equal to 1
    temp = inside * (radii >= wmbm) * (radii <= wpbm) #upper radial transition
    wavelet[temp] *= np.cos(np.pi * ewt_beta(wam * (radii[temp] - wmbm)) / 2)
    temp = inside * (radii >= wmbn) * (radii <= wpbn) #lower radial transition
    wavelet[temp] *= np.sin(np.pi * ewt_beta(wan * (radii[temp] - wmbn)) / 2)
    
    if ang_bound2 - ang_bound1 != np.pi:
        temp = inside * (theta >= mbm) * (theta <= pbm) #upper angular transition
        wavelet[temp] *= np.cos(np.pi * ewt_beta(an * (theta[temp] - mbm)) / 2)
        temp = inside * (theta >= mbn) * (theta <= pbn) #lower angular transition
        wavelet[temp] *= np.sin(np.pi * ewt_beta(an * (theta[temp]-mbn)) / 2)

    return wavelet + wavelet[-1::-1, -1::-1] #symmetrize
    
"""
ppfft(f)
Performs the pseudo-polar fast Fourier transform of image f
Input:
    f       - input image f
Output:
    ppff    - pseudo-polar fourier transform of image f
Author: Basile Hurat""" 
def ppfft(f):
    #f is assumed N x N where N is even. If not, we just force it to be
    [h, w] = f.shape
    N = h
    f2 = f
    if h != w or np.mod(h, 2) == 1:
        N = int(np.ceil(max(h, w) / 2) * 2) #N is biggest dimension, but force even
        f2 = np.zeros([N, N])
        f2[N // 2 - int(h / 2):N // 2 - int(h / 2) + h, N // 2 - int(w / 2):N // 2 - int(w / 2) + w] = f
    ppff = np.zeros([2 * N, 2 * N]) * 1j
    
    #Constructing Quadrants 1 and 3
    ff = np.fft.fft(f2, N * 2, axis=0)
    ff = np.fft.fftshift(ff, 0)
    for i in range(-N, N):
        ppff[i + N, N - 1::-1] = fracfft(ff[i + N, :], i / (N ** 2), 1)
    
    #Constructing quadrants 2 and 4
    ff = np.fft.fft(f2, N * 2, axis=1)
    ff = np.fft.fftshift(ff, 1)
    ff = ff.T
    
    for i in range(-N, N):
        x = np.arange(0, N)
        factor = np.exp(1j * 2 * np.pi * x * (N / 2 - 1) * i / (N ** 2))
        ppff[i + N, N:2 * N] = fracfft(ff[i + N, :] * factor, i / (N ** 2))
    return ppff

"""
fracfft(f)
Performs the fractional fast Fourier transform of image f
Input:
    f           - input image f
    alpha       - fractional value for fractional fft
    centered    - whether or not this is centered
Output:
    result      - fractional fourier transform of image f
Author: Basile Hurat""" 
def fracfft(f, alpha, centered=0):
    f = np.reshape(f.T, f.size)#flatten f
    N = len(f)#get length
    
    if centered == 1:
        x = np.arange(0, N)
        factor = np.exp(1j * np.pi * x * N * alpha)
        f = f * factor
    
    x = np.append(np.arange(0, N), np.arange(-N, 0))
    factor = np.exp(-1j * np.pi * alpha * x ** 2)
    ff = np.append(f, np.zeros(N))
    ff = ff * factor
    XX = np.fft.fft(ff)
    YY = np.fft.fft(np.conj(factor))
    
    result = np.fft.ifft(XX * YY)
    result = result * factor
    result = result[0:N]
    return result

def ippfft(ppff, acc=1e-5):
    [h, w] = ppff.shape
    h = h // 2
    
    w = np.sqrt(np.abs(np.arange(-h, h)) / 2) / h
    w[h + 1] = np.sqrt(1 / 8) / h
    w = np.outer(np.ones(2 * h), w)
    recon = np.zeros([h, h])
    delta = 1
    count = 0
    while delta > acc and count < 1000:
        error = w * (ppfft(recon).T - ppff.T)
        D = appfft(w * error).T
        delta = np.linalg.norm(D)
        mu = 1 / h
        recon = recon - mu * D
        count += 1
    if count == 1000:
        print('could not converge during inverse pseudo-polar fft')
    return recon

def appfft(X):
    [h, w] = X.shape
    h = h // 2
    
    Y = 1j * np.zeros([h, h])
    temp = 1j * np.zeros([h, 2 * h])
    for i in range(-h, h):
        Xvec = X[h - 1::-1, i + h]
        alpha = -i / (h ** 2)
        OneLine = fracfft(Xvec, alpha)
        OneLine = (OneLine.T) * np.exp(-1j * np.pi * i * np.arange(0, h) / h)
        temp[:,i + h] = OneLine
    Temp_Array = 2 * h * np.fft.ifft(temp, axis = 1)
    Temp_Array = Temp_Array[:, 0:h].dot(np.diag(np.power(-1, np.arange(0, h))))
    Y = Temp_Array.T
    
    temp2 = 1j * np.zeros([h, 2 * h])
    for i in range(-h, h):
        Xvec = X[-1:h - 1:-1, i + h]
        alpha = i / (h ** 2)
        OneCol = fracfft(Xvec, alpha)
        OneCol = (OneCol) * (np.exp(1j * np.pi * i * np.arange(0, h) / h).T)
        temp2[:, i + h] = OneCol
    Temp_Array = 2 * h * np.fft.ifft(temp2, axis=1) 
    Y += Temp_Array[:, 0:h].dot(np.diag(np.power(-1, np.arange(0, h))))
    Y = Y.T
    return Y


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
    params = ewt_params()

    [ewtc, mfb, bounds_scales] = ewt2dLP(imagem,params)
    #print(ewtc)
    ewtc=np.asarray(ewtc)
    print(ewtc.shape)
    k=ewtc.shape[0]
    plt.figure(figsize=(15,7))
    for i in range(k):
        print(i)
        plt.subplot(1, k+1, i + 1)
        l=ewtc[i,:,:]
        plt.imshow(ewtc[i,:,:],cmap='gray',vmax=l.max(),vmin=l.min())
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    coeffs = pywt.dwt2(imagem, 'haar')
    cA, (cH, cV, cD) = coeffs 
    plt.figure(figsize=(15,7))
    plt.subplot(1, 4, 1)
    plt.imshow(cA,cmap='gray',vmax=cA.max(),vmin=cA.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.subplot(1, 4, 2)
    plt.imshow(cH,cmap='gray',vmax=cH.max(),vmin=cH.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.subplot(1, 4, 3)
    plt.imshow(cD,cmap='gray',vmax=cD.max(),vmin=cD.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.subplot(1, 4, 4)
    plt.imshow(cV,cmap='gray',vmax=cV.max(),vmin=cV.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    coeffs = pywt.wavedec2(imagem, 'db1')
    print(len(coeffs)-1)
    print(pywt.dwtn_max_level(imagem.shape, 'db1'))
    try:
        eng = matlab.engine.start_matlab()
    
        cwtmorl = eng.cwtft2(imagem)
        eng.quit()
    except:
        pass
    
    
    
def main():
    imagem=modelimage()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




