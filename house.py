"""
Machine Learning: House prices (Linear Regression)
Author : Bappy Ahmed
Email: bappymalik4161@gmail.com
17 sept 2020
"""

#Importing lybraries
import pandas as pd #For reading csv file
import numpy as np
import matplotlib.pyplot as plt #for plotted the data sets in a graph
from sklearn.model_selection import train_test_split  #To split data 
from sklearn.linear_model import LinearRegression  #To train


# Importing data sets
#My data_sets
data_set = pd.read_csv('house price.csv')

# d = data_set.head(3)
# print(d)
# data_set.shape

# #To cheak the null values into the data sets
# # data_set.isnull().any()
# data_set.isnull().sum()

# separate data
# x is the feature so 2 Dimension
x = data_set[['area']]
# y is the label so 1 Dimension
y = data_set['price']


# # Visualization Data
# plt.scatter(data_set['area'], data_set['price'], marker='x', color='red')
# plt.xlabel('Area in square ft')
# plt.ylabel('Price in taka')
# plt.title('Home prices in Dhaka')


# split data
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = .30, random_state = 1)


# Training Data sets
#Create an object
reg = LinearRegression()
reg.fit(xtrain, ytrain)

# predict
r1 = reg.predict(xtest)
r2 = reg.predict([[3300]])  #individual
# print(r2)


# #To draw the best fit line
# plt.scatter(data_set['area'], data_set['price'], marker='x', color='red')
# plt.xlabel('Area in square ft')
# plt.ylabel('Price in taka')
# plt.title('Home prices in Dhaka')
# plt.plot(data_set.area, reg.predict(data_set[['area']]))


"""
# y= mx+c
reg.coef_  #coefficient
reg.intercept_
y = reg.coef_*x + reg.intercept_
print(y)
"""