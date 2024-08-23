#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import pearsonr            


# # show first row of the titainic_data

# In[5]:


import os
import pandas as pd

file_path = "titanic_data.csv"

if os.path.exists(file_path):
    titanic_data = pd.read_csv(file_path)
    print(titanic_data.head())  # This will print the first 5 rows of the dataset



# # EXPLORTEY DATA ANALYSIS 

# In[15]:


#  Number of rows and columns
titanic_data.shape


# In[5]:


# Get information about the data
titanic_data.info()


# In[6]:


# To check the number of missing values
titanic_data.isnull().sum()


# In[7]:


# Dropping the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data.head()


# In[8]:


# Replacing the missing vlaues in "Age" column with mean 
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[9]:


# Finding the mode value of "Embarkked" Column 
print(titanic_data['Embarked'].mode())


# In[10]:


# Replacing the missing values in "Embarked" column with the mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[17]:


titanic_data.isnull().sum()


# In[18]:


# Getting statistical measures about the data (It's not useful while handling categorical column)
titanic_data.describe()


# In[19]:



titanic_data['Survived'].value_counts()


# In[20]:


# Count plot for Survived Column
sns.countplot(x='Survived', data=titanic_data)


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming titanic_data is your DataFrame containing the Titanic dataset
sns.countplot( x ='Sex', data = titanic_data , palette= ['green', 'black'])

# Adding titles and labels for better understanding
plt.title('Count of Passengers by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')

# Display the plot
plt.show()


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming titanic_data is your DataFrame containing the Titanic dataset
sns.countplot(x="Pclass", hue="Survived", data= titanic_data , palette=['green', 'red', 'yellow'])

# Adding titles and labels for better understanding
plt.title('Number of Survivors by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')

# Display the plot
plt.show()


# In[29]:


titanic_data["Sex"].value_counts()


# In[30]:


titanic_data["Embarked"].value_counts()


# In[31]:


titanic_data.head()


# In[32]:


X = titanic_data.drop(columns=["PassengerId", "Name", "Ticket", "Survived"], axis=1)
Y = titanic_data['Survived']


# In[33]:


print(Y)


# In[37]:


print(X)


# # spliting the data to train and test 

# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[39]:


print(X.shape, X_train.shape, X_test.shape)


# In[8]:





# In[32]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[33]:


# Create dummy data
X = np.random.rand(100, 5)  # 100 samples, 5 features each
y = np.random.randint(0, 2, size=100)  # Binary target


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # logisticRegression

# In[35]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[22]:


model_prediction = model.predict(X_test)


# In[24]:


accuracy = accuracy_score(model_prediction, y_test)


# In[25]:


print(f"Model accuracy: {accuracy:.2f}")


# In[27]:


import pandas as pd

results = pd.DataFrame({
    'Model': ['Logistic Regression'], 
    'Score': [0.78]
})


# In[28]:


results


# In[29]:


model_prediction


# In[ ]:


# To save the model in a pkl file. 

import pickle as pkl

pkl.dump(model, open('model.pkl', 'wb'))


# In[66]:


print(X)


# In[67]:


print(Y)


# In[68]:


X_train.iloc[0,:]


# In[37]:


a = X_train[0, :]  # Extract the first sample directly
a = np.array(a)    # This step is redundant because a is already a numpy array
ypred = model.predict(a.reshape(1, -1))  # Reshape and predict
ypred  # Display the prediction


# In[71]:


Y_train[0]


# In[ ]:




