#IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#GET THE DATA
customers = pd.read_csv('Ecommerce Customers')
# print(customers.head())
print(customers.info())
# print(customers.describe())

#EMPLORATORY DATA ANALYSIS
# sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
# sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
# sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')
# sns.pairplot(customers)
# sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
# plt.show()

X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# TRAINING THE MODEL
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)

#PREDICTING TEST DATA
prediction = lm.predict(X_test)
# sns.scatterplot(x=y_test,y=prediction)
# plt.show()

#EVALUATING THE MODEL
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,prediction))
print('MSE:',metrics.mean_squared_error(y_test,prediction))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,prediction)))

#RESIDUALS
# sns.distplot(a=(y_test-prediction),bins=50)
# plt.show()

#Conclusion
coefficients = pd.DataFrame(data=lm.coef_,index=X.columns,columns=['Coefficient'])
print(coefficients)
