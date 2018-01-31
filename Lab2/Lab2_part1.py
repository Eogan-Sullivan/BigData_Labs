import pandas as pd
data = pd.read_csv('C:/Users/t00166087/Desktop/Lab2/lab2_datafiles/Customer Churn Model.txt')
account_length = data['Account Length']
account_length.head()

Subdata = data[['Account Length','VMail Message','Day Calls']]
Subdata.head()


wanted = ['Account Length','VMail Message','Day Calls']
column_list = data.columns.values.tolist()
sublist = [x for x in column_list if x not in wanted]
subdata = data[sublist]
subdata.head()

#How to select rows
data[1:50]
data[25:75]
data[:50]
data[51:]

datal = data[data['Day Mins']>500]
datal.shape

datal = data[data['State'] == 'VA']
datal.shape

datal = data[(data['Day Mins']>500) & (data['State'] == 'VA')]
datal.shape

datal = data[(data['Day Mins']>500) | (data['State'] == 'VA')]
datal.shape

subdata_first_50 = data[['Account Length','VMail Message','Day Calls']][1:50]
subdata_first_50

subdata[1:50]

data.ix[1:100,1:6]

data.ix[:,1:6]

data.ix[1:100,:]

data.ix[1:100,[2,5,7]]

data.ix[[1,2,5],[2,5,7]]

data.ix[[1,2,5],['Area Code','VMail Message','Day Mins']]

data['Total Mins'] = data['Day Mins'] +data['Eve Mins']+data['Night Mins']
data['Total Mins'].head()

#generating random numbers
import numpy as np
np.random.randint(1,100)

import numpy as np
np.random.random()

def randint_range(n,a,b):
    x=[]
    for i in range(n):
        x.append(np.random.randint(a,b))
        return x
    
randint_range(10,2,1000) 

import random
for i in range(3):
    print random.randrange(0,100,5)
    
a = range(100)
np.random.shuffle(a)

c