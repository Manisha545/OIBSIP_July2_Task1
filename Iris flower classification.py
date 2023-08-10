#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np #linear algebra
import pandas as pd #data processing

#Input directory
import os
for dirname, _, filenames in os.walk(r"C:\Users\manis\Downloads\archive"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# In[10]:


df=pd.read_csv(r'C:\Users\manis\Downloads\archive\Iris.csv')


# In[11]:


df.head()


# In[12]:


df.shape


# In[13]:


df.describe()


# In[14]:


df.info


# In[15]:


df.isnull().sum()


# In[16]:


df.Species.value_counts


# In[17]:


X= df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y= df['Species']


# In[18]:


X


# In[19]:


y


# In[20]:


#Do the train/test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# In[21]:


#Training the linear regression model

from sklearn.linear_model import LogisticRegression


# In[22]:


# Let's create an instance for the LogisticRegression model
lr = LogisticRegression()

# Train the model on our train dataset
lr.fit(X,y)

# Train the model with the training set

lr.fit(X_train,y_train)


# In[23]:


# Getting predictions from the model for the given examples.
predictions = lr.predict(X)

# Compare with the actual charges

Scores = pd.DataFrame({'Actual':y,'Predictions':predictions})
Scores.head()


# In[24]:


y_test_hat=lr.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat)*100,'%')


# In[ ]:




