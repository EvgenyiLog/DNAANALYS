#!/usr/bin/env python
# coding: utf-8

# In[8]:


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
    plt.figure('Violinplot',figsize=(15,7))   
    sns.violinplot(data=df.loc[:,'xcentrrect':'perimetersrect'])
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет 
    
    plt.figure('Boxplot',figsize=(15,7))    
    sns.boxplot(df.loc[:,'xcentrrect':
       'perimetersrect'])
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    try:
        plt.figure('Histplot',figsize=(15,7))
        sns.histplot(df.loc[:,'xcentrrect':'perimetersrect'])
        plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    except:
        pass
    try:
        plt.figure('Barplot',figsize=(15,7))  
        sns.barplot(df.loc[:,'xcentrrect':'perimetersrect'])
        plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    except:
        pass
    plt.show()
    
if __name__ == "__main__":
    main()
        


# In[ ]:





# In[ ]:




