import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('KNN_Project_Data')
print(df.head())

# sns.pairplot(data=df,hue='TARGET CLASS')

#Standardize the Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_feat = scaler.transform(df.drop('TARGET CLASS',axis=1))

df_scaled = pd.DataFrame(scaled_feat,columns=df.columns[:-1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_feat,df['TARGET CLASS'],test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

prediction = knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,4))
plt.plot(range(1,40),error_rate,linestyle='--',marker='o',color='blue',markerfacecolor='red',markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K Value')
plt.show()