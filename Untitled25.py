#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#inline: This part of the command specifies that the plots should be displayed directly in the notebook, inline with the code, rather than in a separate external window


# In[36]:


train = pd.read_csv('C:\\Users\\Vanshika Rana\\OneDrive\\Desktop\\titanic dataset\\Titanic-Dataset.csv')


# In[37]:


train.head()


# In[38]:


train.isnull()


# In[39]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[40]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[41]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train)


# In[42]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train)


# In[43]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[44]:


sns.displot(train['Age'].dropna(),kde=False,bins=40) #kde-kernet density function #hist


# In[45]:


sns.countplot(x='SibSp',data=train)


# In[46]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[47]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
        


# In[48]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[49]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[50]:


train.drop('Cabin',axis=1,inplace=True)


# In[51]:


train.head()


# In[52]:


train.info()


# In[53]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[54]:


embark=pd.get_dummies(train['Embarked'], drop_first=True).head()


# In[55]:


sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[56]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[57]:


train.head()


# In[58]:


train=pd.concat([train,sex,embark],axis=1)


# In[59]:


train.head()

train.drop['Survived'].head()
# In[60]:


train.drop(['Survived'],axis=1).head()


# In[61]:


train['Survived'].head()


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(train.drop(['Survived'],axis=1),train['Survived'],test_size=0.30,random_state=101)


# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


import numpy as np

# Check for NaN values in x_train
nan_check_x = np.isnan(x_train)
if np.any(nan_check_x):
    print("There are NaN values in x_train.")

# Check for NaN values in y_train
nan_check_y = np.isnan(y_train)
if np.any(nan_check_y):
    print("There are NaN values in y_train.")

# Check for infinite values in x_train
inf_check_x = np.isinf(x_train)
if np.any(inf_check_x):
    print("There are infinite values in x_train.")

# Check for infinite values in y_train
inf_check_y = np.isinf(y_train)
if np.any(inf_check_y):
    print("There are infinite values in y_train.")


# In[73]:


import numpy as np
x_train.fillna(x_train.mean(), inplace=True)


# In[75]:


x_train


# In[77]:


import numpy as np

# Check for NaN values in x_train
nan_check_x = np.isnan(x_train)
if np.any(nan_check_x):
    print("There are NaN values in x_train.")

# Check for NaN values in y_train
nan_check_y = np.isnan(y_train)
if np.any(nan_check_y):
    print("There are NaN values in y_train.")

# Check for infinite values in x_train
inf_check_x = np.isinf(x_train)
if np.any(inf_check_x):
    print("There are infinite values in x_train.")

# Check for infinite values in y_train
inf_check_y = np.isinf(y_train)
if np.any(inf_check_y):
    print("There are infinite values in y_train.")


# In[78]:


logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)


# In[79]:


logmodel = LogisticRegression(max_iter=1000)  # You can adjust the number of iterations as needed


# In[80]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(x_train_scaled, y_train)


# In[81]:


predictions= logmodel.predict(x_train_scaled)


# In[82]:


from sklearn.metrics import confusion_matrix


# In[85]:


print("y_test shape:", y_test.shape)
print("predictions shape:", predictions.shape)


# In[86]:


conf_matrix = confusion_matrix(y_test, predictions[:268]) 
print("Confusion Matrix:")
print(conf_matrix)


# In[87]:


from sklearn.metrics import accuracy_score


# In[89]:


accuracy=accuracy_score(y_test,predictions[:268])
accuracy


# In[90]:


predictions


# In[ ]:




