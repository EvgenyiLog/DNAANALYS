#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
    else:
        assert print('The file does not exist')
    return data
        
        
def main():
    df=xslxreader(os.path.abspath("C:/Users/evgen/Downloads/contourcentrintensivityrectfilt16parrect.xlsx"))
    plt.figure('Violinplot',figsize=(15,7))   
    sns.violinplot(data=df)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет 
    
    plt.figure('Boxplot',figsize=(15,7))    
    sns.boxplot(df)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    try:
        plt.figure('Histplot',figsize=(15,7))
        sns.histplot(df)
        plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    except:
        pass
    try:
        plt.figure('Barplot',figsize=(15,7))  
        sns.barplot(df)
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




