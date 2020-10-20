'''
** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
(You'll import sklearn as you need it.)**
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**
customers = pd.read_csv('Ecommerce Customers')
# **Check the head of customers, and check out its info() and describe() methods.**
# print(customers.head())
# print(customers.info())
# print(customers.describe())

# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**
# sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)

# ** Do the same but with the Time on App column instead. **
# sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**
# sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')

# **Let's explore these types of relationships across the entire data set
# sns.pairplot(data=customers)

# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **
# sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)
# plt.show()

# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **
X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# Now its time to train our model on our training data!
# ** Import LinearRegression from sklearn.linear_model **
from sklearn.linear_model import LinearRegression

# **Create an instance of a LinearRegression() model named lm.**
lm = LinearRegression()

# ** Train/fit lm on the training data.**
lm.fit(X_train,y_train)

# **Print out the coefficients of the model**
print(lm.coef_)

# ** Use lm.predict() to predict off the X_test set of the data.**
prediction = lm.predict(X_test)

# ** Create a scatterplot of the real test values versus the predicted values. **
# sns.scatterplot(x=y_test,y=prediction)
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')
# plt.show()

# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_true=y_test,y_pred=prediction))
print('MSE:',metrics.mean_squared_error(y_test,prediction))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,prediction)))

# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**
# sns.distplot((y_test-prediction))
# plt.show()

coefficients = pd.DataFrame(data=lm.coef_, index=X.columns,columns=['Coefficient'])
print(coefficients)

