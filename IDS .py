#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
#in order to ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[41]:


# Assuming the dataset is stored in a variable named `data
df=pd.DataFrame({ 'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [19,20,25,22,26,23,20,22,24,21],
    'Gender': ['F', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'F'],
    'Study Hours': [3, 6, 2, 4, 3, 7, 6, 3, 5, 4],
    'Attendance': [70, 80, 90, 100, 100, 90, 80, 70, 80, 85],
    'Exam Score': [80, 70, 90, 85, 80, 78, 82, 88, 86, 92]
    
})


# In[42]:


df.head()


# In[43]:


df.info()


# In[44]:


df.shape


# In[45]:


df.describe


# In[46]:


df['Exam Score'] = df['Exam Score'].apply(lambda x: float(x))


# In[47]:


X = df.drop('Exam Score', axis=1)  # Predictor variables
y = df['Exam Score']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[48]:


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Gender'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)


# In[49]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Gender'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# In[51]:


y_train_pred = regression_model.predict(X_train)
y_test_pred = regression_model.predict(X_test)


# In[52]:


from sklearn.metrics import mean_squared_error, r2_score

# Calculate MSE, RMSE, and R-squared for the training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate MSE, RMSE, and R-squared for the testing set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Print the metrics for the training set
print("Training set:")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R-squared:", train_r2)

# Print the metrics for the testing set
print("Testing set:")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R-squared:", test_r2)


# In[53]:


column_names = []
encoded_columns = ct.transformers_[0][1].get_feature_names_out(['Gender'])
column_names.extend(encoded_columns)
column_names.extend(X.columns[1:])


# In[38]:


coefficients = pd.DataFrame({'Feature': column_names, 'Coefficient': regression_model.coef_})
significant_predictors = coefficients.loc[coefficients['Coefficient'].abs() > 0]

print("Significant Predictors:")
print(significant_predictors)






