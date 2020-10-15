#** Import pandas and read in the Ecommerce Purchases csv file and set it to a DataFrame called ecom. **
import pandas as pd
df = pd.read_csv('Ecommerce Purchases')
#**Check the head of the DataFrame.**
print(df.head())
#** How many rows and columns are there? **
print(df.info())
#** What is the average Purchase Price? **
print(df['Purchase Price'].mean())
#** What were the highest and lowest purchase prices? **
print(df['Purchase Price'].max())
print(df['Purchase Price'].min())
#** How many people have English 'en' as their Language 
#of choice on the website? **
print(df[df['Language'] == 'en'].count())
#** How many people have the job title of "Lawyer" ? **
print(df[df['Job']=='Lawyer'].count())
#** How many people made the purchase during the AM and
#  how many people made the purchase during PM ? **
print(df['AM or PM'].value_counts())
#** What are the 5 most common Job Titles? **
print(df['Job'].value_counts().head(5))
#** Someone made a purchase that came from Lot: "90 WT" , 
# what was the Purchase Price for this transaction? **
print(df[df['Lot']=='90 WT']['Purchase Price'])
#** What is the email of the person with the following 
# Credit Card Number: 4926535242672853 **
print(df[df['Credit Card'] == 4926535242672853]['Email'])
#** How many people have American Express as their Credit
#  Card Provider *and* made a purchase above $95 ?**
print(df[(df['CC Provider'] == 'American Express') & (df['Purchase Price'] > 95)].count())
#** Hard: How many people have a credit card that expires in 2025? **
print(sum(df['CC Exp Date'].apply(lambda x: x[3:]== '25')))
#** Hard: What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...) **
print(df['Email'].apply(lambda x : x.split('@')[-1]).value_counts().head(5))