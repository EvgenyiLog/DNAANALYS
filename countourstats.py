#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
    df=xslxreader(os.path.abspath("C:/Users/evgen/Downloads/contourcentrintensivityrectfilt16parrectwithoutbackgroundparametrsbf.xlsx"))
    fig=plt.figure('Violinplot',figsize=(15,7))
    plt.subplot(221)
    sns.violinplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(222)
    sns.violinplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(223)
    sns.violinplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(224)
    sns.violinplot(data=df.loc[:,'meanbackgroundrect':'pct90foregroundrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/violinplot1.jpg")
    
    
    
    
    fig=plt.figure('Boxplot',figsize=(15,7))
    plt.subplot(221)
    sns.boxplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(222)
    sns.boxplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(223)
    sns.boxplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(224)
    sns.boxplot(data=df.loc[:,'meanbackgroundrect':'pct90foregroundrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/boxplot1.jpg")
    
    try:
        
        fig=plt.figure('Histplot',figsize=(15,7))
        plt.subplot(221)
        sns.histplot(data=df.loc[:,'xcentrrect':'heightrect'], binwidth=500)
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(222)
        sns.histplot(data=df.loc[:,'anglerect':'y2rect'],binwidth=500)
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(223)
        sns.histplot(data=df.loc[:,'x3rect':'perimetersrect'], binwidth=500)
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(224)
        sns.histplot(data=df.loc[:,'meanbackgroundrect':'pct90foregroundrect'], binwidth=500)
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    
        plt.savefig("C:/Users/evgen/Downloads/histplot1.jpg")
    
    except:
        pass
    try:
        plt.figure('Barplot',figsize=(15,7))
        plt.subplot(221)
        sns.barplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(222)
        sns.barplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(223)
        sns.barplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(224)
        sns.barplot(data=df.loc[:,'meanbackgroundrect':'pct90foregroundrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        
        
        plt.savefig("C:/Users/evgen/Downloads/barplot1.jpg")
    
    except:
        pass
    try:
        plt.figure('Boxenplot',figsize=(15,7))
        plt.subplot(221)
        sns.boxenplot(data=df.loc[:,'xcentrrect':'heightrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(222)
        sns.boxenplot(data=df.loc[:,'anglerect':'y2rect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(223)
        sns.boxenplot(data=df.loc[:,'x3rect':'perimetersrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.subplot(224)
        sns.boxenplot(data=df.loc[:,'meanbackgroundrect':'pct90foregroundrect'], orient="h")
        plt.grid(True)
        plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
        plt.savefig("C:/Users/evgen/Downloads/boxenplot1.jpg")
    except:
        pass
    plt.show()
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




