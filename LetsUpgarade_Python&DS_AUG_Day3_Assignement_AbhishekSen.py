#!/usr/bin/env python
# coding: utf-8

# # 1. Generate random data with NumPy for 1000 data points with 2 columns only.

# In[1]:


from math import *
import numpy as np
r=np.random.randn(1000)
r.reshape(500,2)


# # 2. Plot Scatter plot, line plot with that in all variations we covered in the class.

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

x1 = 3 * np.random.rand(10, 1)
y1 = 4 + 3 *x1* np.random.randn(10, 1)
plt.scatter(x1, y1)
plt.title("Scatter plot between $X$ and $Y$ for Assignment-3 by $Abhishek$")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()


# In[2]:


##Changing Colour

plt.scatter(x1, y1, c="green")
plt.title("Green colored scatter plot")
plt.xlabel("$X$ axis")
plt.ylabel("$Y$ axis")
plt.show()


# In[3]:


##Changing Size

plt.scatter(x1, y1, s=70)
plt.title("Increased Size Scatter Plots")
plt.xlabel("$X axis$")
plt.ylabel("$Y axis$")
plt.show()


# In[4]:


##Adding Legend

a = 3 * np.random.rand(10, 1)
b = 4 + 3 *a* np.random.randn(10, 1)

c = 2 * np.random.rand(10, 1)
d = 2 + 4 *a* np.random.randn(10, 1)


dataset = [(a, b), 
           (c, d)]


plt.scatter(dataset[0][0], dataset[0][1], c="orange", s=30, label="From a, b")
plt.scatter(dataset[1][0], dataset[1][1], c="violet", s=100, label="From c, d")
plt.title("Labelled data a-b & c-d")
plt.xlabel("$X axis$")
plt.ylabel("$Y axis$")
plt.legend()
plt.show()


# In[6]:


#Changing the shapes of "dots" (Using markers)

plt.scatter(dataset[0][0], dataset[0][1], c="orange", s=30, label="From a, b",marker="*")
plt.scatter(dataset[1][0], dataset[1][1], c="violet", s=100, label="From c, d",marker="^")
plt.title("Labelled data a-b & c-d")
plt.xlabel("$X axis$")
plt.ylabel("$Y axis$")
plt.legend()
plt.show()


# In[7]:


##Changing the Edge Colours

plt.scatter(dataset[0][0], dataset[0][1], c="orange", s=100, label="From a, b",marker="*",edgecolors="k")
plt.scatter(dataset[1][0], dataset[1][1], c="yellow", s=100, label="From c, d",marker="^",edgecolors="r")
plt.title("Labelled data a-b & c-d")
plt.xlabel("$X axis$")
plt.ylabel("$Y axis$")
plt.legend()
plt.show()


# In[8]:


##Shorthand syntax for scatter plots

plt.plot(dataset[0][0], dataset[0][1], "r*", label="From a, b")
plt.plot(dataset[1][0], dataset[1][1], "g^", label="From c, d")
plt.title("Labelled data")
plt.xlabel("$X axis$")
plt.ylabel("$Y axis$")
plt.legend()
plt.show()


# # Line Plot

# In[15]:


x1 = np.array([0, 1, 2, 3, 4])
y1 = np.array([100, 200, 300, 400, 500])
plt.plot(x1, y1)
plt.title("Line plot")
plt.xticks([0, 1, 2, 3, 4])
plt.yticks([100, 200, 300, 400, 500])
plt.xlabel("$X$ axis")
plt.ylabel("$Y$ axis")
plt.show()


# In[16]:


##Changing the color of the line

x1 = np.array([0, 1, 2, 3, 4])
y1 = np.array([100, 200, 300, 400, 500])
plt.plot(x1, y1, c="g")
plt.title("Line plot")
plt.xticks([0, 1, 2, 3, 4])
plt.yticks([100, 200, 300, 400, 500])
plt.xlabel("$X$ axis")
plt.ylabel("$Y$ axis")
plt.show()


# In[17]:


##Changing the Line Width

x1 = np.array([0, 1, 2, 3, 4])
y1 = np.array([100, 200, 300, 400, 500])
plt.plot(x1, y1, c='g',linewidth=10)
plt.title("Line plot")
plt.xticks([0, 1, 2, 3, 4])
plt.yticks([100, 200, 300, 400, 500])
plt.xlabel("$X$ axis")
plt.ylabel("$Y$ axis")
plt.show()


# In[18]:


#Plotting Two Lines at a time

a = np.array([i*2 for i in range(1, 11)])

b = np.array([i*2 for i in range(1, 11)])

d = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30])


dataset = [(a, b), 
           (a, d)]


plt.plot(dataset[0][0], dataset[0][1], c="red",label="From a, b")
plt.plot(dataset[1][0], dataset[1][1], c="green", label="From c, d")
plt.title("Labelled lines")
plt.xlabel("$X-axis$")
plt.ylabel("$Y-axis$")
plt.legend()
plt.show()


# In[19]:


##Changing Line Types

plt.plot(dataset[0][0], dataset[0][1], c="yellow",label="From a, b", linestyle="dashed")
plt.plot(dataset[1][0], dataset[1][1], c="seagreen", label="From c, d", linestyle="dotted")
plt.title("Labelled lines")
plt.xlabel("$X-axis$")
plt.ylabel("$Y-axis$")
plt.legend()
plt.show()


# In[20]:


##Line Plot with Scatter Plot

plt.scatter(dataset[0][0], dataset[0][1], c="orange",label="From a, b")
plt.plot(dataset[1][0], dataset[1][1], c="seagreen", label="From c, d")
plt.title("Labelled lines")
plt.xlabel("$X-axis$")
plt.ylabel("$Y-axis$")
plt.legend()
plt.show()


# In[21]:


##Subplots

plt.subplot(1, 2, 1)
plt.plot(dataset[0][0], dataset[0][1], c="yellow",label="Yellow Line")
plt.title("YELLOW LINE")
plt.xlabel("$X-axis$")
plt.ylabel("$Y-axis$")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(dataset[1][0], dataset[1][1], c="orange", label="Orange Line")
plt.title("ORANGE LINE")
plt.xlabel("$X-axis$")
plt.ylabel("$Y-axis$")
plt.legend()

plt.subplots_adjust(left=2, right=4)
plt.show()


# # 3. Do data analysis on two features of Boston data and write your insights.

# In[22]:


import plotly.express as px
from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# In[23]:


boston_dataset.DESCR


# In[25]:


sns.displot(boston['AGE'])
plt.show()


# In[27]:


plt.plot(boston["INDUS"],boston["INDUS"], c="green",label="Non-Retail Areas")
plt.scatter(boston["CRIM"],boston["CRIM"], c="red", label="Crime Rate")
plt.title("Labelled lines")
plt.xlabel("$Non-Retail Business Acres/Town$")
plt.ylabel("$Per Capita Crime Rate$")
plt.legend()
plt.show()


# In[29]:


#Observation/Insight: The Crime rate is equally distributed throughout the Non-retail business areas. 
#The 'Crime Rate' is directly proportional to "Non-Retail Business Areas"


# # 4. Apply simple linear regression and submit it as an assignment.

# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print(boston_dataset.keys())


# In[31]:


X = boston_dataset['data']
y = boston_dataset['target']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


# In[33]:


lr = LinearRegression()


# In[34]:


lr.fit(X_train, y_train)


# In[35]:


lr.coef_


# In[36]:


lr.intercept_


# In[37]:


lr.score(X_train, y_train)


# In[38]:


lr.score(X_test,y_test)


# In[39]:


print(f"Original Value from test at 3rd index: ", y_test[2])


# In[40]:


sample = X_test[2]


# In[41]:


lr.predict([sample])


# In[ ]:




