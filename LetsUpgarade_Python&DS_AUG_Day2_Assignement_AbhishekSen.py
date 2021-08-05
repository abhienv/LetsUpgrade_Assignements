#!/usr/bin/env python
# coding: utf-8

# # Question 1
# #Create an Array with five names and print all five in reverse.
# 

# In[1]:


import numpy as np
from array import *
arr_name=["Abhi", "Sunil","Rajat", "Dhruba", "Mani"]
arr_name.reverse()
a=np.array(arr_name)
for i in a:
    print(i)


# # Question 2
# #print only the elements which are divisible by 5 from 1 to 50
# 

# In[2]:


for i in range(1,50):
    if (i%5)==0:
        print (i)


# In[ ]:




