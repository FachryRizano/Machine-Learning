#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the data
df = pd.read_csv('College_Data',index_col=0)
# print(df.head())
# print(df.info())
# print(df.describe())

#Data Visualization
# sns.scatterplot(x='Room.Board',y='Grad.Rate',hue='Private',data=df)

# sns.scatterplot(x='Outstate', y='F.Undergrad', data=df, hue='Private',alpha=0.5)
# print(df['Private'].head())
# g = sns.FacetGrid(data=df,hue='Private',palette='coolwarm',size=6)
# g = g.map(plt.hist,'Outstate',bins=20)

# g = sns.FacetGrid(data=df,hue='Private',palette='coolwarm',aspect=2)
# g = g.map(plt.hist,'Grad.Rate',bins=20)
# plt.tight_layout()
# plt.show()
df['Grad.Rate'].loc['Cazenovia College'] = 100
# print(df['Grad.Rate'].loc['Cazenovia College'])
# g = sns.FacetGrid(data=df,hue='Private',palette='coolwarm',aspect=3)
# g = g.map(plt.hist,'Grad.Rate',bins=20)
# plt.show()

# KMeans Cluster Creation
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
km.fit(df.drop('Private',axis=1))
print(km.cluster_centers_)

def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],km.labels_))
print(classification_report(df['Cluster'],km.labels_))