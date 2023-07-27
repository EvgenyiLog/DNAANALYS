#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt

def genimage(m,n):
    image=np.zeros((m,n))
    image[2,2]=255
    image=np.asarray(image,dtype=np.uint8)
    print(image)
import warnings    
def main():
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    warnings.filterwarnings(action='ignore', category= DeprecationWarning)
    genimage(4,4)
    
    
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




