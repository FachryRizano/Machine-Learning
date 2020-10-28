import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# the label is loan_status

#Function to take the information of features

#Feature Information
df_feat = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
def info(feature):
    print(feature + ' : ' + df_feat.loc[feature]['Description'])


#Load the Data
df = pd.read_csv('lending_club_loan_two.csv')
# print(df.head())
# print(df.info())
# print(df.describe())

#Exploratory Data Analysis
# sns.countplot(x='loan_status',data=df)

# df['loan_amnt'].hist(b)
# plt.show()

# print(df.corr())

#Visualize the correlation
# plt.figure(figsize=(12,7),tight_layout=True)
# sns.heatmap(df.corr(),cmap='viridis',annot=True)
# plt.ylim(10,0)
# plt.show()

'''
that has high correlation is loan_amnt vs installment = 0.95
and total_acc vs open_acc = 0.68
'''
# info('loan_amnt')
# info('installment')

# sns.scatterplot(x='installment',y='loan_amnt',data=df)
# plt.show()

# sns.boxplot(x='loan_status',y='loan_amnt',data=df)
# plt.show()

# print(df.groupby('loan_status')['loan_amnt'].describe())
# print(sorted(df['grade'].unique()))
# print(sorted(df['sub_grade'].unique()))

# sns.countplot(x='grade',data=df,hue='loan_status')
# plt.show()

# plt.figure(figsize=(12,7),tight_layout=True)
# sg_order = sorted(df['sub_grade'].unique())
# sns.countplot(x='sub_grade',data=df,palette='coolwarm',order=sg_order)
# sns.countplot(x='sub_grade',palette='coolwarm',hue='loan_status',order=sg_order, data=df)

# data_f_and_g =df[(df['grade'] == 'F') | (df['grade'] == 'G')]
# sg_order = sorted(data_f_and_g['sub_grade'].unique())
# sns.countplot(x='sub_grade',data=data_f_and_g,hue='loan_status',order = sg_order)
# plt.show()

# print(df['loan_status'].unique())
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off':0})
# print(df[['loan_repaid','loan_status']])

# df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
# plt.show()

'''
DATA PREPROCESSING
'''
# miss_data = pd.Series(100*df.isnull().sum()/len(df))
# print(miss_data)
# info('emp_title')
# info('emp_length')

# print(df['emp_title'].nunique())
# print(df['emp_title'].value_counts())

df = df.drop('emp_title',axis=1)

# el_sort = sorted(df['emp_length'].dropna().unique())
# print(el_sort)
sorted_el = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']
# plt.figure(figsize=(12,7))
# sns.countplot(x='emp_length',data=df,order = sorted_el,palette='rainbow')
# plt.show()

# plt.figure(figsize=(12,7),tight_layout=True)
# sns.countplot(x='emp_length',hue='loan_status',data=df,order=sorted_el)
# plt.show()

# emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
# print(emp_co)
# emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']

# emp_len = emp_co/emp_fp
# emp_len.plot(kind='bar')
# plt.show()
df = df.drop('emp_length',axis=1)

# print(df.isnull().sum())
# print(df[['purpose','title']].head(10))
df = df.drop('title',axis=1)

# info('mort_acc')

# print(df['mort_acc'].value_counts())
# print(df.corr()['mort_acc'].sort_values())

# print('Mean of mort_acc column per total_acc')
# print(df.groupby('total_acc')['mort_acc'].mean())

avg_acc = df.groupby('total_acc')['mort_acc'].mean()

def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return avg_acc[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)
# print(df.isnull().sum())

df = df.dropna()
# print(df.isnull().sum())

'''
CATEGORICAL VARIABLES AND DUMMY VARIABLES
'''

# print(df.select_dtypes('object').columns)
'''
'term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',    
       'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
       'initial_list_status', 'application_type', 'address'
'''

#TERM FEATURE
# print(df['term'].head(5))
# print(df['term'].value_counts())
df['term'] = df['term'].apply(lambda term : int(term[:3]))

#GRADE FEATURE
df = df.drop('grade',axis=1)

subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

# print(df.columns)
# print(df.select_dtypes('object').columns)

dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)
df = df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

#HOME_OWNERSHIP FEATURE
# print(df['home_ownership'].value_counts())
df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
# print(df.columns)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

#ADDRESS FEATURE
df['zip_code'] = df['address'].apply(lambda address: address[-5:])
# print(df['zip_code'].head(5))

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)
# print(df.columns)

#issue_d feature
df = df.drop('issue_d',axis=1)

#EARLIEST_CR_LINE FEATURE
# print(df['earliest_cr_line'].head(5))
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
# print(df['earliest_cr_year'].head(5))
df = df.drop('earliest_cr_line',axis=1)

'''
TRAIN TEST SPLIT
'''
from sklearn.model_selection import train_test_split
df = df.drop('loan_status',axis=1)
# print(df.info())
X = df.drop('loan_repaid',axis=1).values
y= df['loan_repaid'].values

#Grabbing a Sample fro Training Time
df = df.sample(frac=0.1,random_state=101)
print(len(df))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)

'''
NORMALIZING THE DATA
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''
CREATING THE MODEL
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(78,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy')

model.fit(X_train,y_train,batch_size=256,epochs=25,validation_data=(X_test,y_test))
from tensorflow.keras.models import load_model
model.save('full_data_project_model.h5')

'''
EVALUATING MODEL PERFORMANCE
'''

losses = pd.DataFrame(model.history.history,columns=['loss','val_loss'])
# losses.plot()
# plt.show()

prediction = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
