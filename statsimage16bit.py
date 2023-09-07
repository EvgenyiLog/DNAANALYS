#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    return data
        
        
def main():
    df=xslxreader(os.path.abspath("C:/Users/evgen/Downloads/contourcentrintensivityrectfilt16parrectwithoutbackgroundparametrsbf2.xlsx"))
    fig=plt.figure('Boxplot stat rect',figsize=(15,7))
    sns.boxplot(data=df.loc[: ,'meanbackgroundrect':'pct90foregroundrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/boxplot1rect.jpg")
    
    fig=plt.figure('Boxenplot stat',figsize=(15,7))
    sns.boxenplot(data=df.loc[:,'meanbackgroundrect':'pct90foregroundrect'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/boxenplot1rect.jpg")
    
    
    fig=plt.figure('Boxplot rect',figsize=(15,7))
    plt.subplot(121)
    sns.boxplot(data=df['intensivityrect'], orient="v",color='red').set_title('intensivityrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(122)
    sns.boxplot(data=df['areasrect'], orient="v",color='lime').set_title('areasrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет 
    plt.savefig("C:/Users/evgen/Downloads/boxplot2rect.jpg")
    
    fig=plt.figure('Boxenplot rect',figsize=(15,7))
    plt.subplot(121)
    sns.boxenplot(data=df['intensivityrect'], orient="v",color='red').set_title('intensivityrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(122)
    sns.boxenplot(data=df['areasrect'], orient="v",color='lime').set_title('areasrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет 
    plt.savefig("C:/Users/evgen/Downloads/boxenplot2rect.jpg") 
    
    
    fig=plt.figure('Dcentrrect',figsize=(15,7))
    plt.subplot(221)
    sns.histplot(df['dcentrrect'], kde=True,color='red').set_title('dcentrrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(222)
    sns.boxplot(df['dcentrrect'],color='m').set_title('dcentrrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(223)
    sns.violinplot(df['dcentrrect'],color='lime').set_title('dcentrrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(224)
    sns.boxenplot(df['dcentrrect'],color='blue').set_title('dcentrrect',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет 
    plt.savefig("C:/Users/evgen/Downloads/dcentrrect.jpg")
    
    df=xslxreader(os.path.abspath("C:/Users/evgen/Downloads/contourcentrintensivityfilt16parwithoutbackgroundparametrsbf2.xlsx"))
    fig=plt.figure('Boxplot stat',figsize=(15,7))
    sns.boxplot(data=df.loc[: ,'meanbackground':'pct90foreground'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/boxplot1.jpg")
    
    fig=plt.figure('Boxenplot stat',figsize=(15,7))
    sns.boxenplot(data=df.loc[:,'meanbackground':'pct90foreground'], orient="h")
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.savefig("C:/Users/evgen/Downloads/boxenplot1.jpg")
    
    
    fig=plt.figure('Boxplot',figsize=(15,7))
    plt.subplot(121)
    sns.boxplot(data=df['intensivity'], orient="v",color='red').set_title('intensivity',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(122)
    sns.boxplot(data=df['areas'], orient="v",color='lime').set_title('areas',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет 
    plt.savefig("C:/Users/evgen/Downloads/boxplot2.jpg")
    
    fig=plt.figure('Boxenplot',figsize=(15,7))
    plt.subplot(121)
    sns.boxenplot(data=df['intensivity'], orient="v",color='red').set_title('intensivity',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(122)
    sns.boxenplot(data=df['areas'], orient="v",color='lime').set_title('areas',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет 
    plt.savefig("C:/Users/evgen/Downloads/boxenplot2.jpg") 
    
    
    fig=plt.figure('Dcentr',figsize=(15,7))
    plt.subplot(221)
    sns.histplot(df['dcentr'], kde=True,color='red').set_title('dcentr',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(222)
    sns.boxplot(df['dcentr'],color='m').set_title('dcentr',fontsize=10)
    plt.grid(True)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(223)
    sns.violinplot(df['dcentr'],color='lime').set_title('dcentr',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет
    plt.subplot(224)
    sns.boxenplot(df['dcentr'],color='blue').set_title('dcentr',fontsize=10)
    plt.grid(True)
    plt.tick_params(labelsize =10,#  Размер подписи
                    color = 'k')   #  Цвет 
    plt.savefig("C:/Users/evgen/Downloads/dcentr.jpg") 
    
    
    
    
    
    
    
    plt.show()
    
if __name__ == "__main__":
    main()


# In[ ]:




