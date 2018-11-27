
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('customer_churn.csv')
X = dataset.iloc[:,1:6].values
y = dataset['Churn']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the  results
y_pred = classifier.predict(X_test)

# Getting  the Confusion Matrix,Accuracy Score,Classification Report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)
print(cm) 
print(accuracy)
print(classification_report(y_test, y_pred))
