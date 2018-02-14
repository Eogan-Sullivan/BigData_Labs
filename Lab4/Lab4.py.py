
# coding: utf-8

# In[1]:


import pandas as pd
advert=pd.read_csv('C:\\Users\\t00166087\\Desktop\\Lab4\\lab4_datafiles\\Advertising.csv')
advert.head()


# In[2]:


import statsmodels.formula.api as smf
model1=smf.ols(formula='Sales~TV',data=advert).fit()
model1.params


# In[3]:


model1.pvalues


# In[4]:


model1.rsquared


# In[6]:


model1.summary()


# In[7]:


sales_pred=model1.predict(pd.DataFrame(advert['TV']))
sales_pred


# In[8]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
advert.plot(kind='scatter', x='TV', y='Sales')
plt.plot(pd.DataFrame(advert['TV']),sales_pred,c='red',linewidth=2)


# In[10]:


import numpy as np
advert['sales_pred']=0.047537*advert['TV']+7.03
advert['RSE']=(advert['Sales']-advert['sales_pred'])**2
RSEd=advert.sum()['RSE']
RSE=np.sqrt(RSEd/198)
salesmean=np.mean(advert['Sales'])
error=RSE/salesmean
RSE,salesmean,error


# In[11]:


import statsmodels.formula.api as smf
model2=smf.ols(formula='Sales~TV+Newspaper',data=advert).fit()
model2.params


# In[12]:


sales_pred=model2.predict(advert[['TV','Newspaper']])
sales_pred


# In[13]:


import numpy as np
advert['sales_pred']=5.77 + 0.046*advert['TV'] + 0.04*advert['Newspaper']
advert['RSE']=(advert['Sales']-advert['sales_pred'])**2
RSEd=advert.sum()['RSE']
RSE=np.sqrt(RSEd/197)
salesmean=np.mean(advert['Sales'])
error=RSE/salesmean
RSE,salesmean,error


# In[15]:


model2.summary()


# In[16]:


import statsmodels.formula.api as smf
model3=smf.ols(formula='Sales~TV+Radio',data=advert).fit()
model3.params


# In[17]:


import statsmodels.formula.api as smf
model3=smf.ols(formula='Sales~TV+Radio',data=advert).fit()
model3.params


# In[19]:


sales_pred=model3.predict(advert[['TV','Radio']])
sales_pred


# In[20]:


model3.summary()


# In[21]:


import statsmodels.formula.api as smf
model4=smf.ols(formula='Sales~TV+Radio+Newspaper',data=advert).fit()
model4.params


# In[22]:


sales_pred=model4.predict(advert[['TV','Radio','Newspaper']])
sales_pred


# In[23]:


model4.summary()


# In[24]:


import numpy as np
a=np.random.randn(len(advert))
check=a<0.8
training=advert[check]
testing=advert[~check]


# In[25]:


import statsmodels.formula.api as smf
model5=smf.ols(formula='Sales~TV+Radio',data=training).fit()
model5.summary()


# In[26]:


sales_pred=model5.predict(training[['TV','Radio']])
sales_pred


# In[28]:


import numpy as np
testing['sales_pred']=2.86 + 0.04*testing['TV'] + 0.17*testing['Radio']
testing['RSE']=(testing['Sales']-testing['sales_pred'])**2
RSEd=testing.sum()['RSE']
RSE=np.sqrt(RSEd/51)
salesmean=np.mean(testing['Sales'])
error=RSE/salesmean
RSE,salesmean,error


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
feature_cols = ['TV', 'Radio']
X = advert[feature_cols]
Y = advert['Sales']
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
lm = LinearRegression()
lm.fit(trainX, trainY)


# In[30]:


print(lm.intercept_)
print(lm.coef_)


# In[31]:


zip(feature_cols,lm.coef_)


# In[32]:


lm.score(trainX,trainY)


# In[33]:


lm.predict(testX)


# In[34]:


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
feature_cols = ['TV', 'Radio','Newspaper']
X = advert[feature_cols]
Y = advert['Sales']
estimator = SVR(kernel="linear")
selector = RFE(estimator,2,step=1)
selector = selector.fit(X, Y)


# In[35]:


selector.support_


# In[36]:


selector.ranking_


# In[37]:


# Third Lab Sheet In Lab 4 Folder


# In[39]:


import pandas as pd
df=pd.read_csv('C:\\Users\\t00166087\\Desktop\\Lab4\\lab4_datafiles\\Ecom Expense.csv')
df.head()


# In[40]:


dummy_gender=pd.get_dummies(df['Gender'],prefix='Sex')
dummy_city_tier=pd.get_dummies(df['City Tier'],prefix='City')


# In[41]:


dummy_city_tier


# In[44]:


dummy_gender


# In[43]:


column_name=df.columns.values.tolist()
df1=df[column_name].join(dummy_gender)
column_name1=df1.columns.values.tolist()
df2=df1[column_name1].join(dummy_city_tier)
df2


# In[45]:


from sklearn.linear_model import LinearRegression
feature_cols = ['Monthly Income','Transaction Time','City_Tier 1','City_Tier 2','City_Tier 3','Sex_Female','Sex_Male']
X = df2[feature_cols]
Y = df2['Total Spend']
lm = LinearRegression()
lm.fit(X,Y)


# In[47]:


print(lm.intercept_)
print(lm.coef_)
zip(feature_cols, lm.coef_)


# In[48]:


lm.score(X,Y)


# In[49]:


import numpy as np
df2['total_spend_pred']=3720.72940769 + 0.12*df2['Transaction Time']+0.15*df2['Monthly Income']+119*df2['City_Tier 1']-16*df2['City_Tier 2']
-102*df2['City_Tier 3']-94*df2['Sex_Female']+94*df2['Sex_Male']
df2['RSE']=(df2['Total Spend']-df2['total_spend_pred'])**2
RSEd=df2.sum()['RSE']
RSE=np.sqrt(RSEd/2354)
salesmean=np.mean(df2['Total Spend'])
error=RSE/salesmean
RSE,salesmean,error


# In[50]:


dummy_gender=pd.get_dummies(df['Gender'],prefix='Sex').iloc[:, 1:]
dummy_city_tier=pd.get_dummies(df['City Tier'],prefix='City').iloc[:, 1:]
column_name=df.columns.values.tolist()
df3=df[column_name].join(dummy_gender)
column_name1=df3.columns.values.tolist()
df4=df3[column_name1].join(dummy_city_tier)
df4


# In[51]:


from sklearn.linear_model import LinearRegression
feature_cols = ['Monthly Income','Transaction Time','City_Tier 2','City_Tier 3','Sex_Male']
X = df2[feature_cols]
Y = df2['Total Spend']
lm = LinearRegression()
lm.fit(X,Y)


# In[53]:


print(lm.intercept_)
print(lm.coef_)
zip(feature_cols, lm.coef_)


# In[55]:


import pandas as pd
data = pd.read_csv('C:/Users/t00166087/Desktop/Lab4/lab4_datafiles/Auto.csv')
data.head()


# In[56]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
data['mpg']=data['mpg'].dropna()
data['horsepower']=data['horsepower'].dropna()


# In[57]:


plt.plot(data['horsepower'],data['mpg'],'ro')
plt.xlabel('Horsepower')
plt.ylabel('MPG (Miles Per Gallon)')


# In[58]:


import numpy as np
from sklearn.linear_model import LinearRegression
X=data['horsepower'].fillna(data['horsepower'].mean())
Y=data['mpg'].fillna(data['mpg'].mean())
lm=LinearRegression()
lm.fit(X[:,np.newaxis],Y)


# In[59]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(data['horsepower'],data['mpg'],'ro')
plt.plot(X,lm.predict(X[:,np.newaxis]),color='blue')


# In[61]:


lm.score(X[:,np.newaxis],Y)


# In[62]:


RSEd=(Y-lm.predict(X[:,np.newaxis]))**2
RSE=np.sqrt(np.sum(RSEd)/389)
ymean=np.mean(Y)
error=RSE/ymean
RSE,error


# In[63]:


import numpy as np
from sklearn.linear_model import LinearRegression
X=data['horsepower'].fillna(data['horsepower'].mean())*data['horsepower'].fillna(data['horsepower'].mean())
Y=data['mpg'].fillna(data['mpg'].mean())
lm=LinearRegression()
lm.fit(X[:,np.newaxis],Y)


# In[64]:


type(lm.predict(X[:,np.newaxis]))
RSEd=(Y-lm.predict(X[:,np.newaxis]))**2
RSE=np.sqrt(np.sum(RSEd)/390)
ymean=np.mean(Y)
error=RSE/ymean
RSE,error,ymean


# In[65]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
X=data['horsepower'].fillna(data['horsepower'].mean())
Y=data['mpg'].fillna(data['mpg'].mean())
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X[:,np.newaxis])
clf = linear_model.LinearRegression()
clf.fit(X_, Y)


# In[66]:


print (clf.intercept_)
print (clf.coef_)


# In[67]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
X=data['horsepower'].fillna(data['horsepower'].mean())
Y=data['mpg'].fillna(data['mpg'].mean())
poly = PolynomialFeatures(degree=5)
X_ = poly.fit_transform(X[:,np.newaxis])
clf = linear_model.LinearRegression()
clf.fit(X_, Y)


# In[68]:


print(clf.intercept_)
print(clf.coef_)

