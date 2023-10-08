#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from PIL import Image
import opensimplex as simplex
from skimage.morphology import local_maxima
from skimage.feature import peak_local_max
#import pyemd 

#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  https://github.com/laszukdawid/PyEMD/issues
#
# Feel free to contact for any information.

import logging

import numpy as np
from scipy.interpolate import Rbf

try:
    from skimage.morphology import reconstruction
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "EMD2D and BEMD are not supported. Feel free to play around and improve them. "
        + "Required dependencies are in `requirements-extra`."
    )


class BEMD:
    """
    **Bidimensional Empirical Mode Decomposition**
    **Important**: This class intends to be undocumented until it's actually properly tested
    and proven to work. An attempt to replicate findings in the paper cited below has failed.
    This method is only included in the package because someone asked for it, and I'm hoping
    that one day someone else will come and *fix it*. Until then, USE AT YOUR OWN RISK.
    The guess why the decomposition doesn't work is that it's difficult to extrapolate image
    far away from extrema. Not even mirroring helps in this case.
    Method decomposition 2D arrays like gray-scale images into 2D representations of
    Intrinsic Mode Functions (IMFs).
    The algorithm is based on Nunes et. al. [Nunes2003]_ work.
    .. [Nunes2003] J.-C. Nunes, Y. Bouaoune, E. Delechelle, O. Niang, P. Bunel.,
       "Image analysis by bidimensional empirical mode decomposition. Image and Vision Computing",
       Elsevier, 2003, 21 (12), pp.1019-1026.
    """

    logger = logging.getLogger(__name__)

    def __init__(self):
        # ProtoIMF related
        self.mse_thr = 0.01
        self.mean_thr = 0.01

        self.FIXE = 1  # Single iteration by default, otherwise results are terrible
        self.FIXE_H = 0
        self.MAX_ITERATION = 5

    def __call__(self, image, max_imf=-1):
        return self.bemd(image, max_imf=max_imf)

    def extract_max_min_spline(self, image, min_peaks_pos, max_peaks_pos):
        """Calculates top and bottom envelopes for image.
        Parameters
        ----------
        image : numpy 2D array
        Returns
        -------
        min_env : numpy 2D array
            Bottom envelope in form of an image.
        max_env : numpy 2D array
            Top envelope in form of an image.
        """
        xi, yi = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        min_val = np.array([image[x, y] for x, y in zip(*min_peaks_pos)])
        max_val = np.array([image[x, y] for x, y in zip(*max_peaks_pos)])
        min_env = self.spline_points(min_peaks_pos[0], min_peaks_pos[1], min_val, xi, yi)
        max_env = self.spline_points(max_peaks_pos[0], max_peaks_pos[1], max_val, xi, yi)
        return min_env, max_env

    @classmethod
    def spline_points(cls, X, Y, Z, xi, yi):
        """Creates a spline for given set of points.
        Uses Radial-basis function to extrapolate surfaces. It's not the best but gives something.
        Grid data algorithm didn't work.
        """
        spline = Rbf(X, Y, Z, function="cubic")
        return spline(xi, yi)

    @classmethod
    def find_extrema_positions(cls, image):
        """
        Finds extrema, both minima and maxima, based on morphological reconstruction.
        Returns extrema where the first and second elements are x and y positions, respectively.
        Parameters
        ----------
        image : numpy 2D array
            Monochromatic image or any 2D array.
        Returns
        -------
        min_peaks_pos : numpy array
            Minima positions.
        max_peaks_pos : numpy array
            Maxima positions.
        """
        min_peaks_pos = BEMD.extract_minima_positions(image)
        max_peaks_pos = BEMD.extract_maxima_positions(image)
        return min_peaks_pos, max_peaks_pos

    @classmethod
    def extract_minima_positions(cls, image):
        return BEMD.extract_maxima_positions(-image)

    @classmethod
    def extract_maxima_positions(cls, image):
        seed_min = image - 1
        dilated = reconstruction(seed_min, image, method="dilation")
        cleaned_image = image - dilated
        return np.where(cleaned_image > 0)[::-1]

    @classmethod
    def end_condition(cls, image, IMFs):
        """Determines whether decomposition should be stopped.
        Parameters
        ----------
        image : numpy 2D array
            Input image which is decomposed.
        IMFs : numpy 3D array
            Array for which first dimensions relates to respective IMF,
            i.e. (numIMFs, imageX, imageY).
        """
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(image, rec):
            return True

        return False

    def check_proto_imf(self, proto_imf, proto_imf_prev, mean_env):
        """Check whether passed (proto) IMF is actual IMF.
        Current condition is solely based on checking whether the mean is below threshold.
        Parameters
        ----------
        proto_imf : numpy 2D array
            Current iteration of proto IMF.
        proto_imf_prev : numpy 2D array
            Previous iteration of proto IMF.
        mean_env : numpy 2D array
            Local mean computed from top and bottom envelopes.
        Returns
        -------
        boolean
            Whether current proto IMF is actual IMF.
        """
        # TODO: Sifting is very sensitive and subtracting const val can often flip
        #      maxima with minima in decomposition and thus repeating above/below
        #      behaviour. For now, mean_env is checked whether close to zero excluding
        #      its offset.
        if np.all(np.abs(mean_env - mean_env.mean()) < self.mean_thr):
            # if np.all(np.abs(mean_env)<self.mean_thr):
            return True

        # If very little change with sifting
        if np.allclose(proto_imf, proto_imf_prev, rtol=0.01):
            return True

        # If IMF mean close to zero (below threshold)
        if np.mean(np.abs(proto_imf)) < self.mean_thr:
            return True

        # Everything relatively close to 0
        mse_proto_imf = np.mean(proto_imf * proto_imf)
        if mse_proto_imf > self.mse_thr:
            return False

        return False

    def bemd(self, image, max_imf=-1):
        """Performs bidimensional EMD (BEMD) on grey-scale image with specified parameters.
        Parameters
        ----------
        image : numpy 2D array,
            Grey-scale image.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.
        Returns
        -------
        IMFs : numpy 3D array
            Set of IMFs in form of numpy array where the first dimension
            relates to IMF's ordinary number.
        """
        image_s = image.copy()

        imf = np.zeros(image.shape)
        imf_old = imf.copy()

        imfNo = 0
        IMF = np.empty((imfNo,) + image.shape)
        notFinished = True

        while notFinished:
            self.logger.debug("IMF -- " + str(imfNo))

            res = image_s - np.sum(IMF[:imfNo], axis=0)
            imf = res.copy()
            mean_env = np.zeros(image.shape)
            stop_sifting = False

            # Counters
            n = 0  # All iterations for current imf.
            n_h = 0  # counts when mean(proto_imf) < threshold

            while not stop_sifting and n < self.MAX_ITERATION:
                n += 1
                self.logger.debug("Iteration: %i", n)

                min_peaks_pos, max_peaks_pos = self.find_extrema_positions(imf)
                self.logger.debug(
                    "min_peaks_pos = %i  |  max_peaks_pos = %i", len(min_peaks_pos[0]), len(max_peaks_pos[0])
                )
                if len(min_peaks_pos[0]) > 1 and len(max_peaks_pos[0]) > 1:
                    min_env, max_env = self.extract_max_min_spline(imf, min_peaks_pos, max_peaks_pos)
                    mean_env = 0.5 * (min_env + max_env)

                    imf_old = imf.copy()
                    imf = imf - mean_env

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE + 1:
                            stop_sifting = True

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:
                        if n == 1:
                            continue
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            n_h += 1
                        else:
                            n_h = 0

                        # STOP if enough n_h
                        if n_h >= self.FIXE_H:
                            stop_sifting = True

                    # Stops after default stopping criteria are met
                    else:
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            stop_sifting = True

                else:
                    stop_sifting = True

            IMF = np.vstack((IMF, imf.copy()[None, :]))
            imfNo += 1

            if self.end_condition(image, IMF) or (max_imf > 0 and imfNo >= max_imf):
                notFinished = False
                break

        res = image_s - np.sum(IMF[:imfNo], axis=0)
        if not np.allclose(res, 0):
            IMF = np.vstack((IMF, res[None, :]))
            imfNo += 1

        return IMF
    
    
import logging

import numpy as np

try:
    from scipy.interpolate import SmoothBivariateSpline as SBS
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
except ImportError:
    raise ImportError(
        "EMD2D and BEMD are not supported. Feel free to play around and improve them. "
        + "Required depdenecies are in `requriements-extra`."
    )


class EMD2D:
    """
    **Empirical Mode Decomposition** on images.

    **Important** This is an experimental module.
    Experiments performed using this module didn't provide acceptable results,
    either in actual output nor in computation performance. The author is not
    an expert in image processing so it's very likely that the code could
    have been improved. Take your best shot.

    Method decomposes images into 2D representations of loose Intrinsic Mode
    Functions (IMFs).

    The current version of the algorithm detects local extrema, separately
    minima and maxima, and then connects them to create envelopes. These
    are then used to create a mean trend and subtracted from the input.

    Threshold values that control goodness of the decomposition:
        * `mse_thr` --- proto-IMF check whether small mean square error.
        * `mean_thr` --- proto-IMF chekc whether small mean value.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, **config):
        # ProtoIMF related
        self.mse_thr = 0.01
        self.mean_thr = 0.01

        self.FIXE = 0
        self.FIXE_H = 0

        self.MAX_ITERATION = 1000

        # Update based on options
        for key in config.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = config[key]

    def __call__(self, image, max_imf=-1):
        return self.emd(image, max_imf=max_imf)

    def extract_max_min_spline(self, image):
        """Calculates top and bottom envelopes for image.

        Parameters
        ----------
        image : numpy 2D array

        Returns
        -------
        min_env : numpy 2D array
            Bottom envelope in form of an image.
        max_env : numpy 2D array
            Top envelope in form of an image.
        """

        big_image = self.prepare_image(image)
        big_min_peaks, big_max_peaks = self.find_extrema(big_image)

        # Prepare grid for interpolation. Doesn't seem necessary.
        xi = np.arange(image.shape[0], image.shape[0] * 2)
        yi = np.arange(image.shape[1], image.shape[1] * 2)

        big_min_image_val = big_image[big_min_peaks]
        big_max_image_val = big_image[big_max_peaks]
        min_env = self.spline_points(big_min_peaks[0], big_min_peaks[1], big_min_image_val, xi, yi)
        max_env = self.spline_points(big_max_peaks[0], big_max_peaks[1], big_max_image_val, xi, yi)

        return min_env, max_env

    @classmethod
    def prepare_image(cls, image):
        """Prepares image for edge extrapolation.
        Method bloats image by mirroring it along all axes. This turns
        extrapolation on edges into interpolation within bigger image.

        Parameters
        ----------
        image : numpy 2D array
            Image for which interpolation is required,

        Returns
        -------
        image : numpy 2D array
            Big image based on the input. Grid 3x3 where the center block is input and
            neighbouring panels are respective mirror images.
        """

        # TODO: This is nasty. Instead of bloating whole image and then trying to
        #      find all extrema, it's better to deal directly with indices.
        shape = image.shape
        big_image = np.zeros((shape[0] * 3, shape[1] * 3))

        image_lr = np.fliplr(image)
        image_ud = np.flipud(image)
        image_ud_lr = np.flipud(image_lr)
        image_lr_ud = np.fliplr(image_ud)

        # Fill center with default image
        big_image[shape[0] : 2 * shape[0], shape[1] : 2 * shape[1]] = image

        # Fill left center
        big_image[shape[0] : 2 * shape[0], : shape[1]] = image_lr

        # Fill right center
        big_image[shape[0] : 2 * shape[0], 2 * shape[1] :] = image_lr

        # Fill center top
        big_image[: shape[0], shape[1] : shape[1] * 2] = image_ud

        # Fill center bottom
        big_image[2 * shape[0] :, shape[1] : 2 * shape[1]] = image_ud

        # Fill left top
        big_image[: shape[0], : shape[1]] = image_ud_lr

        # Fill left bottom
        big_image[2 * shape[0] :, : shape[1]] = image_ud_lr

        # Fill right top
        big_image[: shape[0], 2 * shape[1] :] = image_lr_ud

        # Fill right bottom
        big_image[2 * shape[0] :, 2 * shape[1] :] = image_lr_ud

        return big_image

    @classmethod
    def spline_points(cls, X, Y, Z, xi, yi):
        """Interpolates for given set of points"""

        # SBS requires at least m=(kx+1)*(ky+1) points,
        # where kx=ky=3 (default) is the degree of bivariate spline.
        # Thus, if less than 16=(3+1)*(3+1) points, adjust kx & ky.
        spline = SBS(X, Y, Z)

        return spline(xi, yi)

    @classmethod
    def find_extrema(cls, image):
        """
        Finds extrema, both mininma and maxima, based on local maximum filter.
        Returns extrema in form of two rows, where the first and second are
        positions of x and y, respectively.

        Parameters
        ----------
        image : numpy 2D array
            Monochromatic image or any 2D array.

        Returns
        -------
        min_peaks : numpy array
            Minima positions.
        max_peaks : numpy array
            Maxima positions.
        """

        # define an 3x3 neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_min = maximum_filter(-image, footprint=neighborhood) == -image
        local_max = maximum_filter(image, footprint=neighborhood) == image

        # can't distinguish between background zero and filter zero
        background = image == 0

        # appear along the bg border (artifact of the local max filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        min_peaks = local_min ^ eroded_background
        max_peaks = local_max ^ eroded_background

        min_peaks = local_min
        max_peaks = local_max
        min_peaks[[0, -1], :] = False
        min_peaks[:, [0, -1]] = False
        max_peaks[[0, -1], :] = False
        max_peaks[:, [0, -1]] = False

        min_peaks = np.nonzero(min_peaks)
        max_peaks = np.nonzero(max_peaks)

        return min_peaks, max_peaks

    @classmethod
    def end_condition(cls, image, IMFs):
        """Determins whether decomposition should be stopped.

        Parameters
        ----------
        image : numpy 2D array
            Input image which is decomposed.
        IMFs : numpy 3D array
            Array for which first dimensions relates to respective IMF,
            i.e. (numIMFs, imageX, imageY).
        """
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(image, rec):
            return True

        return False

    def check_proto_imf(self, proto_imf, proto_imf_prev, mean_env):
        """Check whether passed (proto) IMF is actual IMF.
        Current condition is solely based on checking whether the mean is below threshold.

        Parameters
        ----------
        proto_imf : numpy 2D array
            Current iteration of proto IMF.
        proto_imf_prev : numpy 2D array
            Previous iteration of proto IMF.
        mean_env : numpy 2D array
            Local mean computed from top and bottom envelopes.

        Returns
        -------
        boolean
            Whether current proto IMF is actual IMF.
        """

        # TODO: Sifiting is very sensitive and subtracting const val can often flip
        #      maxima with minima in decompoisition and thus repeating above/below
        #      behaviour. For now, mean_env is checked whether close to zero excluding
        #      its offset.
        if np.all(np.abs(mean_env - mean_env.mean()) < self.mean_thr):
            # if np.all(np.abs(mean_env)<self.mean_thr):
            return True

        # If very little change with sifting
        if np.allclose(proto_imf, proto_imf_prev):
            return True

        # If IMF mean close to zero (below threshold)
        if np.mean(np.abs(proto_imf)) < self.mean_thr:
            return True

        # Everything relatively close to 0
        mse_proto_imf = np.mean(proto_imf * proto_imf)
        if mse_proto_imf < self.mse_thr:
            return True

        return False

    def emd(self, image, max_imf=-1):
        """Performs EMD on input image with specified parameters.

        Parameters
        ----------
        image : numpy 2D array,
            Image which will be decomposed.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.

        Returns
        -------
        IMFs : numpy 3D array
            Set of IMFs in form of numpy array where the first dimension
            relates to IMF's ordinary number.
        """
        image_min, image_max = np.min(image), np.max(image)
        offset = image_min
        scale = image_max - image_min

        image_s = (image - offset) / scale

        imf = np.zeros(image.shape)
        imf_old = imf.copy()

        imfNo = 0
        IMF = np.empty((imfNo,) + image.shape)
        notFinished = True

        while notFinished:
            self.logger.debug("IMF -- " + str(imfNo))

            res = image_s - np.sum(IMF[:imfNo], axis=0)
            imf = res.copy()
            mean_env = np.zeros(image.shape)
            stop_sifting = False

            # Counters
            n = 0  # All iterations for current imf.
            n_h = 0  # counts when mean(proto_imf) < threshold

            while not stop_sifting and n < self.MAX_ITERATION:
                n += 1
                self.logger.debug("Iteration: " + str(n))

                min_peaks, max_peaks = self.find_extrema(imf)

                self.logger.debug("min_peaks = %i  |  max_peaks = %i", len(min_peaks[0]), len(max_peaks[0]))
                if len(min_peaks[0]) > 4 and len(max_peaks[0]) > 4:
                    imf_old = imf.copy()
                    imf = imf - mean_env

                    min_env, max_env = self.extract_max_min_spline(imf)

                    mean_env = 0.5 * (min_env + max_env)

                    imf_old = imf.copy()
                    imf = imf - mean_env

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE + 1:
                            stop_sifting = True

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:
                        if n == 1:
                            continue
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            n_h += 1
                        else:
                            n_h = 0

                        # STOP if enough n_h
                        if n_h >= self.FIXE_H:
                            stop_sifting = True

                    # Stops after default stopping criteria are met
                    else:
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            stop_sifting = True

                else:
                    notFinished = False
                    stop_sifting = True

            IMF = np.vstack((IMF, imf.copy()[None, :]))
            imfNo += 1

            if self.end_condition(image, IMF) or (max_imf > 0 and imfNo >= max_imf):
                notFinished = False
                break

        res = image_s - np.sum(IMF[:imfNo], axis=0)
        if not np.allclose(res, 0):
            IMF = np.vstack((IMF, res[None, :]))
            imfNo += 1

        IMF = IMF * scale
        IMF[-1] += offset
        return IMF

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
           
    plt.figure(figsize=(15,7))
    plt.imshow(imagem,cmap='gray',vmax=imagem.max(),vmin=imagem.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений   
    
    fig=plt.figure(figsize=(15,7))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xx,yy,imagem)
    ax.grid(True)
    ay = fig.add_subplot(122)
    ay.imshow(imagem,cmap='gray',vmax=imagem.max(),vmin=imagem.min())
    #ay.grid(True)
    ay.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет делений
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
    x,y=local_maxima(imagem, indices=True)
    print(len(x)/(K1+K2+K3+K4+K5))
    coordinates = peak_local_max(imagem, min_distance=1)
    print(len(coordinates[:,1])/(K1+K2+K3+K4+K5))
    
    bemd = BEMD()
    IMFs = bemd.bemd(imagem, max_imf=10)
    imfNo = IMFs.shape[0]
    print(IMFs.shape)
    print("Done")
    plt.figure(figsize=(15,7))
    for n, imf in enumerate(IMFs):
        print(n)
        plt.subplot(1, imfNo+1, n + 2)
        plt.imshow(imf,cmap='gray',vmax=imf.max(),vmin=imf.min())
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        
        
    emd2d = EMD2D()
    # emd2d.FIXE_H = 5
    IMFs = emd2d.emd(imagem, max_imf=10)
    imfNo = IMFs.shape[0]
    print(IMFs.shape)
    print("Done")
    plt.figure(figsize=(15,7))
    for n, imf in enumerate(IMFs):
        print(n)
        plt.subplot(1, imfNo+1, n + 2)
        plt.imshow(imf,cmap='gray',vmax=imf.max(),vmin=imf.min())
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
def main():
    imagem=modelimage()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()
    
    


# In[ ]:





# In[ ]:





# In[ ]:




