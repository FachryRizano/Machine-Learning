#Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Get the data
loans = pd.read_csv('loan_data.csv')
print(loans.info())
# print(loans.head())
# print(loans.tail())
# print(loans.describe())

#Exploratory Data Analysis
# loans[loans['credit.policy'] == 1]['fico'].hist(bins=30,alpha=0.5,color='blue',label='Credit.policy = 1')
# loans[loans['credit.policy'] == 0]['fico'].hist(bins=30,alpha=0.5,color='red',label='Credit.policy = 0')
# plt.xlabel('FICO')
# plt.legend()
# plt.show()

# loans[loans['not.fully.paid']==1]['fico'].hist(bins=30,alpha=0.5,color='blue',label='not.fully.paid=1')
# loans[loans['not.fully.paid']==0]['fico'].hist(bins=30,alpha=0.5,color='red',label='not.fully.paid=0')
# plt.xlabel('FICO')
# plt.legend()

# sns.countplot(x='purpose',data=loans,hue='not.fully.paid')
# plt.show()

# sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
# plt.show()

# sns.lmplot(x='fico',y='int.rate',hue='credit.policy',col='not.fully.paid',data=loans)
# plt.show()

# SETTING UP THE DATA
# print(loans.info())
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#TRAINING A DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

prediction = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

#Training the Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))