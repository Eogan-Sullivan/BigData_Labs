# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:05:45 2018

@author: t00166087
"""
import pandas as pd
import numpy as np
data = pd.read_csv('C:/Users/t00166087/Desktop/BigData_Labs/Lab2/lab2_datafiles/Customer Churn Model.txt')
column_list = data.columns.values.tolist()

np.random.choice(column_list)

np.random.seed(1)
for i in range(5):
    print (np.random.random())

for i in range(5):
    print(np.random.random())
    

randnum = np.random.uniform(1,100,100)

import matplotlib.pyplot as plt
a= np.random.uniform(1,100,100)
b = range(1,101)
plt.hist(a)

a=np.random.randn(100)
b = range(1,101)
plt.plot(b,a)

a=np.random.randn(100000)
b=range(101)
plt.hist(a)

def pi_run(nums,loops):
    pi_avg = 0
    pi_value_list=[]
    for i in range(loops):
        value = 0
        x = np.random.uniform(0,1,nums).tolist()
        y = np.random.uniform(0,1,nums).tolist()
        for j in range(nums):
            z = np.sqrt(x[j]*x[j]+y[j]*y[j])
            if z<= 1:
                value +=1
        float_value =float(value)
        pi_value =float_value*4/nums
        pi_value_list.append(pi_value)
        pi_avg += pi_value
    
    pi = pi_avg/loops
    ind = range(1,loops+1)
    fig = plt.plot(ind,pi_value_list)
    return(pi,fig)
    
pi_run(1000,100)

d=pd.DataFrame({'A':np.random.randn(10),'B':2.5*np.random.randn(10)+1.5})
d

column_list = data.columns.values.tolist()
a=len(column_list)
d=pd.DataFrame({'Column_Name':column_list,'A':np.random.rand(a),'B':2.5*np.random.randn(a)+1.5})
d

d=pd.DataFrame({'A':np.random.randn(10),'B':2.5*np.random.randn(10)+1.5},index=range(10,20))
d

a=['Male','Female']
b =['Rich','Poor','Middle Class']
gender = []
seb =[]
for i in range(1,101):
    gender.append(np.random.choice(a))
    seb.append(np.random.choice(b))
height=30*np.random.randn(100)+155
weight=20*np.random.randn(100)+60
age = 10*np.random.randn(100)+35
income=1500*np.random.randn(100)+15000

df=pd.DataFrame({'Gender':gender,'Height':height,'Weight':weight,'Age':age,'Income':income,'Socio-Eco':seb})
df.head

df.groupby('Gender')
grouped =df.groupby('Gender')
grouped.groups

grouped = df.groupby('Gender')
for names,groups in grouped:
    print(names)
    print(groups)
    
grouped.get_group('Female')

grouped= df.groupby(['Gender','Socio-Eco'])
for names,groups in grouped:
    print(names)
    print(groups)
    
len(grouped)

grouped= df.groupby(['Gender','Socio-Eco'])
grouped.sum()
grouped.size()
grouped.describe()

grouped_income = grouped['Income']

grouped.aggregate({'Income':np.sum,'Age':np.mean,'Height':np.std})

grouped.aggregate({'Age':np.mean,'Height':lambda x:np.mean(x)/np.std(x)})

grouped.aggregate([np.sum,np.mean,np.std])

grouped['Age'].filter(lambda x:x.sum()>700)

zscore = lambda x:(x-x.mean()/x.std())
grouped.transform(zscore)

f=lambda x:x.fillna(x.mean())
grouped.transform(f)

grouped.head(1)

grouped.tail(1)

grouped=df.groupby('Gender')
grouped.nth(1)

df1 =df.sort_values(['Age','Income'])
grouped = df1.groupby('Gender')
grouped.head(1)
grouped.tail(1)

len(data)
a=np.random.randn(len(data))
check=a<0.8
training = data[check]
testing = data[~check]
len(training)
len(testing)

from sklearn.cross_validation import train_test_split
train,test=train_test_split(data,test_size = 0.2)

import numpy as np
with open(('C:/Users/t00166087/Desktop/BigData_Labs/Lab2/lab2_datafiles/Customer Churn Model.txt','rb') as f:
    data =f.read().split('\n')
np.random.shuffle(data)
train_data = data[:3*len(data)/4]
test_data = data[len(data)/4:]