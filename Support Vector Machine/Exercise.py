import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get the data
iris = sns.load_dataset('iris')

# Exploratory data analysis
# print(iris.head())
# print(iris.info())
# sns.pairplot(iris,hue='species')
# plt.show()

# setosa = iris[iris['species']=='setosa']
# sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True,shade_lowest=False)
# plt.show()

# Train test split
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Train a model
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)

# Model Evaluation
pred = svc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Gridsearch Practise
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
GSV = GridSearchCV(SVC(),param_grid,verbose=2)
GSV.fit(X_train,y_train)
pred_grid = GSV.predict(X_test)
print(confusion_matrix(y_test,pred_grid))
print(classification_report(y_test,pred_grid))
