# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:05:45 2018

@author: t00166087
"""
import pandas as pd
import numpy as np
data = pd.read_csv('C:/Users/t00166087/Desktop/Lab2/lab2_datafiles/Customer Churn Model.txt')
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
