
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


obs = np.random.random(90).reshape(30,3)


# In[5]:


obs


# In[7]:


c1 = np.random.choice(range(len(obs)))
c2 = np.random.choice(range(len(obs)))
clust_cen = np.vstack([obs[c1],obs[c2]])
clust_cen


# In[9]:


from scipy.cluster.vq import vq
vq(obs,clust_cen)


# In[10]:


from scipy.cluster.vq import kmeans
kmeans(obs,clust_cen)


# In[11]:


from scipy.cluster.vq import kmeans
kmeans(obs,2)


# In[27]:


import pandas as pd
df=pd.read_csv('C:\\Users\\t00166087\\Desktop\\Lab5\\lab5_datafiles\\wine.csv',sep=';')
df.head()


# In[13]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.hist(df['quality'])


# In[14]:


df.groupby('quality').mean()


# In[15]:


df_norm = (df - df.min()) / (df.max() - df.min())
df_norm.head()


# In[16]:


from sklearn.cluster import AgglomerativeClustering
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(df_norm)
md=pd.Series(ward.labels_)


# In[17]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.hist(md)
plt.title('Histogram of Cluster Label')
plt.xlabel('Cluster')
plt.ylabel('Frequency')


# In[18]:


ward.children_


# In[19]:


from sklearn.cluster import KMeans
from sklearn import datasets
model=KMeans(n_clusters=6)
model.fit(df_norm)


# In[20]:


model.labels_


# In[21]:


md=pd.Series(model.labels_)
df_norm['clust']=md
df_norm.head()


# In[22]:


model.cluster_centers_


# In[23]:


model.inertia_


# In[24]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.hist(df_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')


# In[25]:


df_norm.groupby('clust').mean()

