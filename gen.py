#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from scipy import signal
import rampy
from sklearn.decomposition import PCA,FastICA

def datreader(f):
    if os.path.splitext(f)[1]==".dat":
        data=pd.read_fwf(f)
        #print(data)
        
        
        
        return data
    
    

        
        
def main():
    data=datreader("C:/Users/evgen/Downloads/pGEM-2_A9.srd.dat")
    #исходные
    data.plot(figsize=(25,7))
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    #data.diff().plot(figsize=(25,7))
    #plt.grid(True)
    #plt.tick_params(labelsize =20,#  Размер подписи
                    #color = 'k')   #  Цвет делений
    
    
    data= data.apply(lambda x: signal.medfilt(x,55))
    data.plot(figsize=(25,7))
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    data=data-data.min()
    data.plot(figsize=(25,7))
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    
    #data.diff().plot(figsize=(25,7))
    #plt.grid(True)
    #plt.tick_params(labelsize =20,#  Размер подписи
                    #color = 'k')   #  Цвет делений
    print(data.max(axis=1))
    print(data.shape[1])
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)

    
    
    # Reformat and view results
    loadings = pd.DataFrame(pca.components_.T,
    columns=['PC%s' % _ for _ in range(len(data.columns))],
    index=data.columns)
    print(loadings)
    
    #plt.figure(figsize=(25,7))
    #plt.plot(pca.explained_variance_ratio_)
    #plt.ylabel('Explained Variance',fontsize=20)
    #plt.xlabel('Components',fontsize=20)
    #plt.grid(True)
    #plt.tick_params(labelsize =20,#  Размер подписи
                    #color = 'k')   #  Цвет делений
    print(data.cov())
    print(data.corr())
    
    datapca=pd.DataFrame(pca.fit_transform(data))
    
    
    datapca.plot(figsize=(25,7))
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    print(data.shape)
    fastica = FastICA(n_components=data.shape[1])
    fastica.fit(data)

    
    
    # Reformat and view results
    loadings = pd.DataFrame(fastica.components_.T,
    columns=['IC%s' % _ for _ in range(len(data.columns))],
    index=data.columns)
    print(loadings)
    
    datafastica=pd.DataFrame(fastica.fit_transform(data))
    
    
    datafastica.plot(figsize=(25,7))
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    
    

    plt.show()
    
    
        
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




