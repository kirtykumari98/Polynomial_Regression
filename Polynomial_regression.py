#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataest
df=pd.read_csv('Position_Salaries.csv')
X=df.iloc[:,1:2].values
y=df.iloc[:,2].values

#Building the Linear Regression model
from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(X,y)

#Bulding the Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_regressor=PolynomialFeatures(degree=4)
poly_X=poly_regressor.fit_transform(X)
poly_regressor.fit(poly_X,y)
linear_regressor2=LinearRegression()
linear_regressor2.fit(poly_X,y)

#Visualising Linear Regression model
plt.scatter(X,y,color='red')
plt.plot(X,linear_regressor.predict(X),color='blue')
plt.title('Linear Regressor')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial Regression Model
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,linear_regressor2.predict(poly_regressor.fit_transform(X_grid)),color='blue')
plt.title('Polynomial Regressor')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
