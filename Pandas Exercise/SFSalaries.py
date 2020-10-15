#** Import pandas as pd.**
import pandas as pd
#** Read Salaries.csv as a dataframe called sal.**
df = pd.read_csv('Salaries.csv')
#** Check the head of the DataFrame. **
print(df.head())
#** Use the .info() method to find out how many entries there are.**
print(df.info())
#**What is the average BasePay ?**
print(df['BasePay'].mean())
#** What is the highest amount of OvertimePay in the dataset ? **
print(df['OvertimePay'].max())
'''** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get 
an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **
'''
print(df[df['EmployeeName']== 'JOSEPH DRISCOLL']['JobTitle'])
#** How much does JOSEPH DRISCOLL make (including benefits)? **
print(df[df['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits'])
#** What is the name of highest paid person (including benefits)
print(df.iloc[df['TotalPayBenefits'].argmax()])
'''** What is the name of lowest paid person (including benefits)? Do you notice something 
strange about how much he or she is paid?**'''
print(df.iloc[df['TotalPayBenefits'].argmin()])
#** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **
print(df.groupby('Year').mean()['BasePay'])
#** How many unique job titles are there? **
print(df['JobTitle'].nunique())
#** What are the top 5 most common jobs? **
print(df['JobTitle'].value_counts().head(5))
#** How many Job Titles were represented by only one person in 2013? 
# (e.g. Job Titles with only one occurence in 2013?) **
print(sum(df[df['Year']== 2013]['JobTitle'].value_counts()==1))
#** How many people have the word Chief in their job title? (This is pretty tricky) **
def chief_string(string):
    if 'chief' in string.lower().split():
        return True
    else:
        return False

print(sum(df['JobTitle'].apply(lambda x:chief_string(x))))
#** Bonus: Is there a correlation between length of the Job Title string and Salary? **
df['title_len'] = df['JobTitle'].apply(len)
print(df[['title_len','TotalPayBenefits']].corr())