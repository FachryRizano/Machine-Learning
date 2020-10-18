# ** Import numpy and pandas **
import numpy as np
import pandas as pd

# ** Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# ** Read in the csv file as a dataframe called df **
df = pd.read_csv('911.csv')
# ** Check the info() of the df **
# print(df.info())

# ** Check the head of df **
# print(df.head())

#BASIC QUESTIONS

# ** What are the top 5 zipcodes for 911 calls? **
# print(df['zip'].value_counts().head(5))

# ** What are the top 5 townships (twp) for 911 calls? **
# print(df['twp'].value_counts().head(5))

# ** Take a look at the 'title' column, how many unique
#  title codes are there? **
# print(df['title'].nunique())

#Creating new features
'''
** In the titles column there are "Reasons/Departments" 
specified before the title code. These are EMS, Fire, and 
Traffic. Use .apply() with a custom lambda expression to 
create a new column called "Reason" that contains this string
 value.** 
**For example, if the title column value is EMS: BACK 
PAINS/INJURY , the Reason column value would be EMS. **
'''
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

# ** What is the most common Reason for a 911 call 
# based off of this new column? **
# print(df['Reason'].value_counts())

# ** Now use seaborn to create a countplot of 911 calls
#  by Reason. **
# sns.countplot(x='Reason',data=df)
# plt.show()

# ** Now let us begin to focus on time information. 
# What is the data type of the objects in the 
# timeStamp column? **
# print(type(df['timeStamp'].iloc[0]))

# ** You should have seen that these timestamps are still
#  strings. Use [pd.to_datetime](http://pandas.pydata.org/
# pandas-docs/stable/generated/pandas.to_datetime.html) to
#  convert the column from strings to DateTime objects. **
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

'''
** You can now grab specific attributes from a Datetime
object by calling them. For example:**
    time = df['timeStamp'].iloc[0]
    time.hour
**You can use Jupyter's tab method to explore
the various attributes you can call. Now that the
timestamp column are actually DateTime objects, 
use .apply() to create 3 new columns called Hour, Month,
and Day of Week. You will create these columns based off 
of the timeStamp column, reference the solutions if you 
get stuck on this step.***
'''
# df['Hour'] = df['timeStamp'].apply(lambda tanggal: tanggal.hour)
# df['Month'] = df['timeStamp'].apply(lambda tanggal:tanggal.month)
# df['Day of Week'] = df['timeStamp'].apply(lambda tanggal:tanggal.dayofweek)

# ** Notice how the Day of Week is an integer 0-6. 
# Use the .map() with this dictionary to map the actual 
# string names to the day of the week: **
# dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:
# 'Sat',6:'Sun'}