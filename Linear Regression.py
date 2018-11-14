# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:43:41 2018

@author: Rahul
"""


#Importing the Library

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Get the dataset
dataset=pd.read_csv('weight-height.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Devide data into training data and test data

from sklearn.model_selection import train_test_split
x_traindata,x_testdata,y_traindata,y_testdata=train_test_split(x,y,test_size=0.25,random_state=0)

# Importing the Linear Regression Model Class from Scikit Learn Library and fit it in our data

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_traindata,y_traindata)

#Predicting the test result 
y_pred = lin_reg.predict(x_testdata)

#Diplay the results by Matplotlib Library

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg.predict((x_grid)),color='blue')
plt.title('Linear Regression')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
