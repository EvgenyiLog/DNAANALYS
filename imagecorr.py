#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


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


def readimage(path):
    'path to file'
    'чтение файла возвращает изображение'
    
    image = cv2.imread(path,0)
    #print(image.shape)
    print(image.dtype)
    return image

def tiffreader(path):
    image=cv2.imread(path,-1)
    #print(image.shape)
    print(image.dtype)
    return image

def corrimage(image1,image2):
    corr = cv2.matchTemplate(image1,image2,cv2.TM_CCOEFF)
    
    print(corr.dtype)
    print(np.amax(corr))
    print(np.amin(corr))
    print(corr.shape)
    corr=cv2.normalize(corr, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_ccoef.jpg",corr)
    corr = cv2.matchTemplate(image1,image2,cv2.TM_CCORR)
    
    print(corr.dtype)
    print(np.amax(corr))
    print(np.amin(corr))
    corr=cv2.normalize(corr, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_ccor.jpg",corr)
    corr = cv2.matchTemplate(image1,image2,cv2.TM_SQDIFF)
    
    print(corr.dtype)
    print(np.amax(corr))
    print(np.amin(corr))
    corr=cv2.normalize(corr, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_sdqif.jpg",corr)
    
    corr = cv2.matchTemplate(image1,image2,cv2.TM_CCOEFF_NORMED)
    
    print(corr.dtype)
    print(np.amax(corr))
    print(np.amin(corr))
    print(corr.shape)
    corr=cv2.normalize(corr, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_ccoef_n.jpg",corr)
    corr = cv2.matchTemplate(image1,image2,cv2.TM_CCORR_NORMED)
    
    print(corr.dtype)
    print(np.amax(corr))
    print(np.amin(corr))
    corr=cv2.normalize(corr, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_ccor_n.jpg",corr)
    corr = cv2.matchTemplate(image1,image2,cv2.TM_SQDIFF_NORMED)
    
    print(corr.dtype)
    print(np.amax(corr))
    print(np.amin(corr))
    corr=cv2.normalize(corr, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_sqdiff_n.jpg",corr)
    
    
    image=cv2.absdiff(image1,image2)
    image=cv2.normalize(image, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    cv2.imwrite("C:/Users/evgen/Downloads/s_1_1102_c_a_absiff.jpg",image)
    
    
    
    
    
def main():
    image1=tiffreader("C:/Users/evgen/Downloads/s_1_1102_c.tif")
    image2=tiffreader("C:/Users/evgen/Downloads/s_1_1102_a.tif")
    image1=readimage("C:/Users/evgen/Downloads/s_1_1102_c.jpg")
    image2=readimage("C:/Users/evgen/Downloads/s_1_1102_a.jpg")
    corrimage(image1,image2)
    plt.show()
    
    
if __name__ == "__main__":
    main()


# In[ ]:



