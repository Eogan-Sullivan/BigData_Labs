
# coding: utf-8

# In[10]:


get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

mtcars = pd.read_csv("C:\\Users\\t00166087\\Desktop\\Tutorial\\mtcars.csv")
mtcars.head()


# In[11]:


bp = plt.boxplot(mtcars[disp])


# In[13]:


disp = mtcars['disp']


# In[14]:


print(disp)


# In[15]:


hp = mtcars['hp']


# In[17]:


weight = mtcars['wt']


# In[18]:


mtList = [disp,hp,weight]


# In[19]:


bp = plt.boxplot(mtList)


# In[31]:


bp = plt.boxplot(disp)


# In[30]:


bp = plt.boxplot(weight)


# In[33]:



bp = plt.boxplot(hp)


# In[34]:


bp = plt.boxplot(weight)


# In[36]:


qsec = mtcars['qsec']


# In[37]:


bp = plt.boxplot(qsec)


# In[39]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
matplotlib.style.use('ggplot')
mtcars.plot(kind="scatter",
x="wt",
y="mpg",
figsize=(9,9),
color="black")


# In[40]:


from sklearn import linear_model
# Initialize model
regression_model = linear_model.LinearRegression()
# Train the model using the mtcars data
regression_model.fit(X = pd.DataFrame(mtcars["wt"]),
y = mtcars["mpg"])
# Check trained model y-intercept
print(regression_model.intercept_)
# Check trained model coefficients
print(regression_model.coef_)


# In[41]:


regression_model.score(X = pd.DataFrame(mtcars["wt"]),
y = mtcars["mpg"])


# In[42]:


train_prediction = regression_model.predict(X = pd.DataFrame(mtcars["wt"]))


# In[43]:


# Actual - prediction = residuals
residuals = mtcars["mpg"] - train_prediction


# In[44]:


residuals.describe()


# In[45]:


SSResiduals = (residuals**2).sum()
SSTotal = ((mtcars["mpg"] - mtcars["mpg"].mean())**2).sum()
# R-squared
1 - (SSResiduals/SSTotal)


# In[46]:


mtcars.plot(kind="scatter",
x="wt",
y="mpg",
figsize=(9,9),
color="black",
xlim = (0,7))
# Plot regression line
plt.plot(mtcars["wt"], # Explanitory variable
train_prediction, # Predicted values
color="blue")


# In[47]:


mtcars_subset = mtcars[["mpg","wt"]]
super_car = pd.DataFrame({"mpg":50,"wt":10}, index=["super"])
new_cars = mtcars_subset.append(super_car)
# Initialize model
regression_model = linear_model.LinearRegression()
# Train the model using the new_cars data
regression_model.fit(X = pd.DataFrame(new_cars["wt"]),
y = new_cars["mpg"])
train_prediction2 = regression_model.predict(X = pd.DataFrame(new_cars["wt"]))
# Plot the new model
new_cars.plot(kind="scatter",
x="wt",
y="mpg",
figsize=(9,9),
color="black", xlim=(1,11), ylim=(10,52))
# Plot regression line
plt.plot(new_cars["wt"], # Explanatory variable
train_prediction2, # Predicted values
color="blue")


# In[49]:


plt.figure(figsize=(9,9))
stats.probplot(residuals, dist="norm", plot=plt)


# In[52]:


from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(train_prediction, mtcars["mpg"])**0.5
RMSE

