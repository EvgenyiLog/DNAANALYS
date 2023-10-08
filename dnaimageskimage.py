#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import color, feature, filters, measure, morphology, segmentation, util

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
#import sunkit_image

import numpy as np

def boxplot_2d(x,y, ax, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k'
    )
    
    
def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    return np.sqrt((s2 - s**2 / ns) / ns)

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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
def imageplot3d(image):
    #x = np.linspace(0, image.shape[1], image.shape[1])
    #y = np.linspace(0, image.shape[0], image.shape[0])
    # full coordinate arrays
    #xx, yy = np.meshgrid(x, y)
    xx, yy = np.ogrid[0:image.shape[0],0:image.shape[1]]
    fig = plt.figure(figsize=(15,7))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs = xx, ys = yy, zs = image)
    ax.plot_surface(xx,yy,image)
    ax.grid(True)
    
    

    
from prettytable import PrettyTable
from colorama import init, Fore, Back, Style     
def localstdmean(image,N):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    noise=np.sqrt((s2 - s**2 / ns) / ns)
    noise=np.asarray(noise,dtype=np.uint8)
    plt.figure(figsize=(15, 7))
    plt.imshow(noise[0:1000,0:1000], cmap=plt.cm.gray,vmax=noise.max(),vmin=noise.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    noises=cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_noise.jpg",noises)
    background=scipy.signal.convolve2d(im, kernel, mode="same")
    background=np.asarray(background,dtype=np.uint8)
    backgrounds=cv2.cvtColor( background, cv2.COLOR_GRAY2BGR) 
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_background.jpg",backgrounds)
    plt.figure(figsize=(15, 7))
    plt.imshow(background[0:1000,0:1000], cmap=plt.cm.gray,vmax=background.max(),vmin=background.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    print(f'image min ={image.min()}')
    print(f'image max ={image.max()}')
    print(f'image mean ={image.mean()}')
    print(f'image std ={image.std()}')
    
    
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def normcorr(image1,image2):
    d = 1

    correlation = np.zeros_like(image1)
    sh_row, sh_col = image1.shape
    for i in range(d, sh_row - (d + 1)):
        for j in range(d, sh_col - (d + 1)):
            correlation[i, j] = correlation_coefficient(image1[i - d: i + d + 1,j - d: j + d + 1],image2[i - d: i + d + 1,j - d: j + d + 1])

    correlation=np.asarray(correlation,dtype=np.uint8)
    plt.figure(figsize=(15,7))
    plt.imshow(correlation[0:1000,0:1000],cmap='gray',vmax=correlation.max(),vmin=correlation.min())
    #plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    try:
        r=cv2.cvtColor(correlation, cv2.COLOR_GRAY2BGR)
    except:
        r=correlation
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_normxcorr.jpg",r)     

def imageground(image):
    # Вычисляем маску фона
    image=np.uint8(image)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Убираем шум
    kernel = np.ones((1, 1), np.uint16)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=50)
    background=opening
    #backgrounds=cv2.cvtColor( background, cv2.COLOR_GRAY2BGR)
    background=cv2.normalize(background, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_background1.jpg",background)
    plt.figure(figsize=(15, 7))
    plt.imshow(background[0:1000,0:1000], cmap=plt.cm.gray,vmax=background.max(),vmin=background.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    

    # создаем маску фона
    sure_bg = cv2.dilate(opening, kernel, iterations=20) 


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  

    foreground=sure_fg
    #cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_foreground.jpg",foreground)
    foreground=cv2.normalize(foreground, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_foreground.jpg",foreground)
    plt.figure(figsize=(15, 7))
    plt.imshow(foreground[0:1000,0:1000], cmap=plt.cm.gray,vmax=foreground.max(),vmin=foreground.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    unknown = cv2.subtract(background,foreground)
    unknown=cv2.normalize(unknown, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_unknown.jpg",unknown)
    plt.figure(figsize=(15, 7))
    plt.imshow(unknown[0:1000,0:1000], cmap=plt.cm.gray,vmax=unknown.max(),vmin=unknown.min())
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    return  background,foreground

    
    
import statistics   
def groundimage(image):
    # Вычисляем маску фона
    image=np.uint8(image)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Убираем шум
    kernel = np.ones((1, 1), np.uint16)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    background=opening
    
    background=cv2.normalize(background, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U) 
    # создаем маску фона
    sure_bg = cv2.dilate(opening, kernel, iterations=2) 


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  

    foreground=sure_fg
    
    foreground=cv2.normalize(foreground, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    unknown = cv2.subtract(background,foreground)
    unknown=cv2.normalize(unknown, None, 0, 4096, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    #print(background.dtype)
    #print(foreground.dtype)
    #print(type(background))
    #print(type(foreground))
    return np.mean(background),np.mean(foreground),np.amax(background),np.amax(foreground),statistics.mode(np.resize(
    background,background.size)),statistics.mode(np.resize(foreground,foreground.size)),np.percentile(
    background,90),np.percentile(foreground,90),np.percentile(background,99),np.percentile(foreground,99)



def main():
    image1=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    image=image1[0:200,0:200]
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    
    #####################################################################
    # Another way of visualizing a grayscale image is contour plotting:

    fig, ax = plt.subplots(figsize=(5, 5))
    qcs = ax.contour(image, origin='image')
    ax.set_title('Contour plot of the same raw image')
    # sphinx_gallery_thumbnail_number = 2


    #####################################################################
    # The contour lines are drawn at these levels:

    qcs.levels

    #####################################################################
    # Each level has, respectively, the following number of segments:

    [len(seg) for seg in qcs.allsegs]

    #####################################################################
    # Estimate the mitotic index
    # ==========================
    # Cell biology uses the
    # `mitotic index <https://en.wikipedia.org/wiki/Mitotic_index>`_
    # to quantify cell division and,
    # hence, cell proliferation. By definition, it is the ratio of cells in
    # mitosis over the total number of cells. To analyze the above image,
    # we are thus interested in two thresholds: one separating the nuclei from the
    # background, the other separating the dividing nuclei (brighter spots) from
    # the non-dividing nuclei.
    # To separate these three different classes of pixels, we
    # resort to :ref:`sphx_glr_auto_examples_segmentation_plot_multiotsu.py`.

    thresholds = filters.threshold_multiotsu(image, classes=3)
    regions = np.digitize(image, bins=thresholds)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(regions)
    ax[1].set_title('Multi-Otsu thresholding')
    ax[1].axis('off')


    #####################################################################
    # Since there are overlapping nuclei, thresholding is not enough to segment
    # all the nuclei.
    # If it were, we could readily compute a mitotic index for this
    # sample:

    cells = image > thresholds[0]
    dividing = image > thresholds[1]
    labeled_cells = measure.label(cells)
    labeled_dividing = measure.label(dividing)
    naive_mi = labeled_dividing.max() / labeled_cells.max()
    print(naive_mi)

    #####################################################################
    # Whoa, this can't be! The number of dividing nuclei

    print(labeled_dividing.max())

    #####################################################################
    # is overestimated, while the total number of cells
    print(labeled_cells.max())

    #####################################################################
    # is underestimated.

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[2].imshow(cells)
    ax[2].set_title('All nuclei?')
    ax[2].axis('off')
    ax[1].imshow(dividing)
    ax[1].set_title('Dividing nuclei?')
    ax[1].axis('off')


    #####################################################################
    # Count dividing nuclei
    # =====================
    # Clearly, not all connected regions in the middle plot are dividing nuclei.
    # On one hand, the second threshold (value of ``thresholds[1]``) appears to be
    # too low to separate those very bright areas corresponding to dividing nuclei
    # from relatively bright pixels otherwise present in many nuclei. On the other
    # hand, we want a smoother image, removing small spurious objects and,
    # possibly, merging clusters of neighboring objects (some could correspond to
    # two nuclei emerging from one cell division). In a way, the segmentation
    # challenge we are facing with dividing nuclei is the opposite of that with
    # (touching) cells.

    #####################################################################
    # To find suitable values for thresholds and filtering parameters, we proceed
    # by dichotomy, visually and manually.

    higher_threshold = 100
    dividing = image > higher_threshold

    smoother_dividing = filters.rank.mean(util.img_as_ubyte(dividing),
                                      morphology.disk(2))

    binary_smoother_dividing = smoother_dividing > 10

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(binary_smoother_dividing)
    ax.set_title('Dividing nuclei')
    ax.axis('off')


    #####################################################################
    # We are left with
    cleaned_dividing = measure.label(binary_smoother_dividing)
    print(cleaned_dividing.max())

    #####################################################################
    # dividing nuclei in this sample.

    #####################################################################
    # Segment nuclei
    # ==============
    # To separate overlapping nuclei, we resort to
    # :ref:`sphx_glr_auto_examples_segmentation_plot_watershed.py`.
    # To visualize the segmentation conveniently, we colour-code the labelled
    # regions using the `color.label2rgb` function, specifying the background            
    # label with argument `bg_label=0`.
 
    distance = ndi.distance_transform_edt(cells)

    local_max_coords = feature.peak_local_max(distance, min_distance=7)                                            
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(cells, cmap='gray')
    ax[0].set_title('Overlapping nuclei')
    ax[0].axis('off')
    ax[1].imshow(color.label2rgb(segmented_cells, bg_label=0))
    ax[1].set_title('Segmented nuclei')
    ax[1].axis('off')


    #####################################################################
    # Additionally, we may use function `color.label2rgb` to overlay the original
    # image with the segmentation result, using transparency (alpha parameter).

    color_labels = color.label2rgb(segmented_cells, image, alpha=0.4, bg_label=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(color_labels)
    ax.set_title('Segmentation result over raw image')
    ##################################################################
    # Finally, we find a total number of

    print(segmented_cells.max())

    #####################################################################
    # cells in this sample. Therefore, we estimate the mitotic index to be:

    print(cleaned_dividing.max() / segmented_cells.max())
    plt.show()
    
    
if __name__ == "__main__":
    main()
    
    
    


# In[ ]:




