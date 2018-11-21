# -*- coding: utf-8 -*-
"""

@author: Rahul
"""
#MULTIPLE LINEAR REGRESSION

#importing the library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Getting the dataset 
dataset=pd.read_csv('Ecommerce_Customers.csv')
x=dataset.iloc[:,3:7].values
y=dataset.iloc[:,7].values

#Divide the data into train and test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Fit The model on the dataset

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predict the Values
y_pred=regressor.predict(x_test)

#Get the Summary of Model 

import statsmodels.formula.api as sm
regressor_ols=sm.OLS(endog=y,exog=x).fit()
regressor_ols.summary()


