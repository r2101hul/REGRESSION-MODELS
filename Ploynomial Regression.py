# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:49:05 2018

@author: Rahul
"""
# Importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Get the Dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Fitting the Ploynomial Regression Model to the Data set 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_ploy=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_ploy,y)

#Predict the New Result at Level 7.5
lin_reg_2.predict(poly_reg.fit_transform(7.5))

#Display The Results
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('Polynomial  Regression')
plt.xlabel('Salary')
plt.ylabel('Position')
plt.show()
