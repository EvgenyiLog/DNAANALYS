#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


def xslxreader(f):
    if os.path.splitext(f)[1]==".xlsx":
        data=pd.read_excel(f)
        #print(data)
        #print(data.values)
        print(data.columns)
    else:
        assert print('The file does not exist')
    return data
        
        
def main():
    df=xslxreader(os.path.abspath("C:/Users/evgen/Downloads/contourcentrintensivityrectfilt16parrect.xlsx"))
    
    fig=plt.figure('Violinplot',figsize=(15,7))
    plt.subplot(131)
    sns.violinplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(132)
    sns.violinplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(133)
    sns.violinplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/violinplot.jpg")
    
    
    
    
    fig=plt.figure('Boxplot',figsize=(15,7))
    plt.subplot(131)
    sns.boxplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(132)
    sns.boxplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(133)
    sns.boxplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/boxplot.jpg")
    
    try:
        
        fig=plt.figure('Histplot',figsize=(15,7))
        plt.subplot(131)
        sns.histplot(data=df.loc[:,'xcentrrect':'heightrect'], binwidth=500)
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(132)
        sns.histplot(data=df.loc[:,'anglerect':'y2rect'],binwidth=500)
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(133)
        sns.histplot(data=df.loc[:,'x3rect':'perimetersrect'], binwidth=500)
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    
        plt.savefig("C:/Users/evgen/Downloads/histplot.jpg")
    
    except:
        pass
    try:
        plt.figure('Barplot',figsize=(15,7))
        plt.subplot(131)
        sns.barplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(132)
        sns.barplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(133)
        sns.barplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        
        
        plt.savefig("C:/Users/evgen/Downloads/barplot.jpg")
    
    except:
        pass
    try:
        plt.figure('Boxenplot',figsize=(15,7))
        plt.subplot(131)
        sns.boxenplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(132)
        sns.boxenplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(133)
        sns.boxenplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.savefig("C:/Users/evgen/Downloads/boxenplot.jpg")
    except:
        pass
        
    plt.show()
    
if __name__ == "__main__":
    main()
        


# In[ ]:





# In[ ]:




