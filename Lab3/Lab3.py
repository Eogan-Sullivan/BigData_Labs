# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:08:37 2018

@author: t00166087
"""

import numpy as np
x =[2,5,7,9,10,15,16]
np.mean(x)
np.median(x)

from sets import Set
y=[2,5,2,8,2,10]
max(Set(y), key=y.count)

#range(max-min)
np.ptp(x)

#variance
np.var(x)

#stamdard deviation
np.std(x)

from numpy import random
x = random.random_integers(5, size = 20)
x

%matplotlib inline
from matplotlib import pyplot as plt
plt.hist(x)
plt.title('Freq Dist')
plt.xlabel('Integer')
plt.ylabel('Value')

from numpy import random
x = random.random_integers(10, size = 100)
x

%matplotlib inline
from matplotlib import pyplot as plt
plt.hist(x)
plt.title('Freq Dist')
plt.xlabel('Integer')
plt.ylabel('Value')

y= random.normal(size=1000)
plt.hist(y)
plt.title('Freq Dist')
plt.xlabel('Integer')
plt.ylabel('Value')

#population of data
import csv
f = open('C:\\Users\\t00166087\\Desktop\\Lab3\\lab3_datafiles\\heightWeight.csv','r')
rdF = csv.reader(f)

hgt = []
wgt = []

for row in rdF:
    hgt.append(row[0])
    wgt.append(row[1])
    
print(hgt)
print(wgt)

height = []
weight =[]
for x in range(len(hgt)):
    height.append(int(hgt[x]))
    weight.append(int(wgt[x]))
    
print(height)
print(weight)

#correlation
np.corrcoef(height,weight)

#hypothesis Testing
a=[10,12,9,11,11,12,9,11,9,9]
b = [13,11,9,12,12,11,12,12,10,11]

from scipy import stats
c = stats.ttest_ind(a,b)
c
d = [13,12,9,12,12,13,12,13,10,11]
c = stats.ttest_ind(a,d)
c

#Box Whisker Plot
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

sensorDF = pd.read_excel("C:\\Users\\t00166087\\Desktop\\Lab3\\lab3_datafiles\\sensors.xlsx")
sensorDF.head()

sensors = list(sensorDF.values.flatten())
sensors

 s1 =sensors[0:12]
s2 = sensors[12:24]
s3 = sensors[24:36]
s4 = sensors[36:48]
print(s1)
print(s2)
print(s3)
print(s4)
sList =[s1,s2,s3,s4]

bp = plt.boxplot(sList)
fig,ax1 = plt.subplots(figsize =(10,6))
ax1.set_title('Comparing Sensor Readings')
ax1.set_xlabel('Sensors')
ax1.set_ylabel('Readings')
bp = ax1.boxplot(sList)
