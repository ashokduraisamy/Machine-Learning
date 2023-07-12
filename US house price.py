#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

#visualisation
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#sns.set_style('whitegrid')

# Model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor


#perfomance
from sklearn.metrics import mean_squared_error,r2_score


# In[2]:


df=pd.read_csv(r'D:\USA_Housing.csv')


# # EDA

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.drop('Address',axis=1,inplace=True)


# In[6]:


df.isna().sum()


# In[7]:


df.describe()


# In[8]:


df.shape


# In[9]:


df.var()


# In[10]:


df.kurt()


# In[11]:


df.skew()


# In[12]:


df.hist(bins=200,figsize=[20,10])


# In[13]:


df.columns


# In[14]:


plt.figure(figsize=(20,8),dpi=400)
sns.boxplot(data=df[[ 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms']],orient='v')
plt.show()


# In[15]:


plt.figure(figsize=(20,8),dpi=400)
sns.boxplot(data=df[[ 'Avg. Area Income', 'Area Population']],orient='v')
plt.show()


# In[16]:


# plt.figure(figsize=(20,8),dpi=400)
sns.boxplot(data=df[[ 'Price']],orient='h')
plt.show()


# # pip install feature_engine

# In[17]:


pip install feature_engine


# In[18]:


from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Avg. Area House Age'])

df['Avg. Area House Age'] = winsor.fit_transform(df[['Avg. Area House Age']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Avg. Area Number of Rooms'])

df['Avg. Area Number of Rooms'] = winsor.fit_transform(df[['Avg. Area Number of Rooms']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Avg. Area Income'])

df['Avg. Area Income'] = winsor.fit_transform(df[['Avg. Area Income']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Area Population'])

df['Area Population'] = winsor.fit_transform(df[['Area Population']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Price'])

df['Price'] = winsor.fit_transform(df[['Price']])


# # Boxplot after dealing with Outliers

# In[19]:


plt.figure(figsize=(20,8),dpi=400)
sns.boxplot(data=df[[ 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms']],orient='h')
plt.show()


# In[20]:


plt.figure(figsize=(20,8),dpi=400)
sns.boxplot(data=df[[ 'Avg. Area Income', 'Area Population']],orient='h')
plt.show()


# In[21]:


plt.figure(figsize=(20,8),dpi=400)
sns.boxplot(data=df[[ 'Price']],orient='h')
plt.show()


# In[22]:


df.to_csv('usehouse_price.csv')


# # #Auto EDA and Visualisation

# In[23]:


pip install dataprep


# In[24]:


from dataprep.eda import create_report
create_report(df)


# In[25]:


pip install AutoViz


# In[26]:


from autoviz.AutoViz_Class import AutoViz_Class   

AV = AutoViz_Class() #instantiaze the AV         

get_ipython().run_line_magic('matplotlib', 'inline')
filename = r'C:\Users\Asus-2022\usehouse_price.csv'
sep = ","
dft = AV.AutoViz(
    filename,
    sep=sep,
    depVar="",
    dfte=None,
    header=0,
    verbose=2,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=2000,
    max_cols_analyzed=20,
)


# # Model Preparation

# In[27]:


X = df.drop(['Price'],axis=1)
y = df['Price']
print(X.shape)


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Linear Regression

# In[29]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# # Use K Fold cross validation to measure accuracy of our LinearRegression model

# In[30]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# # Find best model using GridSearchCV 

# In[31]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


# In[54]:


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])



# In[55]:


find_best_model_using_gridsearchcv(X,y)


# # # Support Vector Regressor

# In[60]:


from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)


# In[61]:


svr.score(X_test,y_test)


# # Random Forest Regressor

# In[62]:


# Model
rfr = RandomForestRegressor(max_depth=10, random_state=42)
rfr.fit(X_train, y_train)
rfr.score(X_test, y_test)


# # XGB Regression

# In[64]:


from xgboost import XGBRegressor


# In[65]:


xgb = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
xgb.fit(X_train,y_train)
xgb.score(X_test,y_test)


# In[ ]:




