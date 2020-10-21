import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')
print(ad_data.tail())
print(ad_data.info())
# print(ad_data.describe())

# sns.histplot(ad_data['Age'],bins=50)

# sns.jointplot(x='Age',y='Area Income',data=ad_data)

# sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')

# sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data, color='green')

# sns.pairplot(ad_data,hue='Clicked on Ad')
# sns.heatmap(data=ad_data.isnull(),cbar=False,yticklabels=False,cmap='viridis')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# data yang bisa digunakan adalah Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, Male, Clicked on Ad
# dependent variable yang ingin dipredict adalah clicked on ad
# independent variable yang ingin digunakan adalah  
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

lm = LogisticRegression()
lm.fit(X_train,y_train)

prediction = lm.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))
