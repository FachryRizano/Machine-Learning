from pandas_datareader import data, wb
import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
# API KEY for alphavantage = X4HTODM34VELLGJC

'''
We need to get data using pandas datareader. We will get stock information for the following banks:
*  Bank of America
* CitiGroup
* Goldman Sachs
* JPMorgan Chase
* Morgan Stanley
* Wells Fargo

** Figure out how to get the stock data from Jan 1st 2006 to Jan 1st 2016 for each of these banks. Set each bank to be a separate dataframe, with the variable name for that bank being its ticker symbol. This will involve a few steps:**
1. Use datetime to set start and end datetime objects.
2. Figure out the ticker symbol for each bank.
2. Figure out how to use datareader to grab info on the stock.
'''

start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)

#Bank of America
BAC = data.DataReader('BAC','av-daily',start,end,api_key = 'X4HTODM34VELLGJC')

#Citigroup
C = data.DataReader('C','av-daily',start,end,api_key = 'X4HTODM34VELLGJC')

#Goldman Sachs
GS = data.DataReader('GS','av-daily',start,end,api_key = 'X4HTODM34VELLGJC')

# * JPMorgan Chase
JPM = data.DataReader('JPM','av-daily',start,end,api_key = 'X4HTODM34VELLGJC')

# * Morgan Stanley
MS = data.DataReader('MS','av-daily',start,end,api_key = 'X4HTODM34VELLGJC')

# * Wells Fargo
WFC = data.DataReader('WFC','av-daily',start,end,api_key = 'X4HTODM34VELLGJC')

# ** Create a list of the ticker symbols (as strings)
# in alphabetical order. Call this list: tickers**
tickers = ['BAC','C','GS','JPM','MS','WFC']

# ** Use pd.concat to concatenate the bank dataframes
# together to a single data frame called bank_stocks. 
# Set the keys argument equal to the tickers list. 
# Also pay attention to what axis you concatenate on.**
bank_stocks = pd.concat(objs=[BAC,C,GS,JPM,MS,WFC],axis=1,keys=tickers)

# ** Set the column name levels (this is filled out for you):**
bank_stocks.columns.names = ['Bank Ticker','Stock Info']

# ** Check the head of the bank_stocks dataframe.**
# print(bank_stocks.head())

#**What is the max Close price for each bank's stock
#throughout the time period?**** What is the max Close 
#price for each bank's stock throughout the time period?**
# print(bank_stocks.xs(key='close',axis=1,level='Stock Info').max())

#  Create a new empty DataFrame called returns
returns = pd.DataFrame()

# ** We can use pandas pct_change() method on the Close 
#column to create a column representing this return value.
# Create a for loop that goes and for each Bank Stock Ticker
#creates this returns column and set's it as a column in 
# the returns DataFrame.**
for tick in tickers:
    returns[tick + 'Return'] = bank_stocks[tick]['close'].pct_change()
# print(returns.head())

# ** Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?**
sns.pairplot(returns[1:])
plt.show()

# ** Using this returns DataFrame, 
# figure out on what dates each bank stock had the best and worst single day returns. 
# You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?**
# print(returns.argmin())
# print(returns.argmax())

# ** Take a look at the standard deviation of the returns,
#  which stock would you classify as the riskiest over the entire time period? 
print(returns.std())
# Which would you classify as the riskiest for the year 2015?**
print(returns.iloc['2015-01-01':'2015-12-31'].std())

