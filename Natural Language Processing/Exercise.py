import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The data
yelp = pd.read_csv('yelp.csv')
# print(yelp.head())
# print(yelp.info())
# print(yelp.describe())

# **Create a new column called "text length" 
# which is the number of words in the text column.**
yelp['text length'] = yelp['text'].apply(len)

# Explore the data
# g = sns.FacetGrid(data=yelp,col='stars',size=6,aspect=2)
# g = g.map(plt.hist,'text length')
# plt.show()

# sns.boxplot(x='stars',y='text length', data = yelp)
# plt.show()

# sns.countplot(x='stars',data=yelp)
# plt.show()

# print(yelp.groupby('stars').mean())
# print(yelp.groupby('stars').mean().corr())

# sns.heatmap(yelp.groupby('stars').mean().corr(),annot=True,cmap='coolwarm')
# plt.show()

# NLP Classification Task

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]
# print(yelp_class.head())

X = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# Training a model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

#Predictions and Evaluations
pred = nb.predict(X_test)

#Eval
from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred))

# Using Text Processing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('Bag of Words',CountVectorizer()),
    ('TFidf',TfidfTransformer()),
    ('classification',MultinomialNB())
])

# re-train test split
Xp = yelp_class['text']
yp = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(Xp,yp,test_size=0.3)
pipeline.fit(X_train,y_train)

#Predictions and Evaluation with Pipeline
pred_pipe = pipeline.predict(X_test)
print(confusion_matrix(y_test,pred_pipe))
print(classification_report(y_test,pred_pipe))
