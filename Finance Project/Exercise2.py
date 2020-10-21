import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_pickle('all_banks')


tickers = ['BAC','C','GS','JPM','MS','WFC']
print(df.xs(key='Close',axis=1,level='Stock Info').max())

returns = pd.DataFrame()
for tick in tickers:
    returns[tick + ' Return'] = df[tick]['Close'].pct_change()
print(returns.head())
sns.pairplot(returns[1:])
plt.show()
print(returns.idxmin())
print(returns.idxmax())
print(returns.std())
print(returns.loc['2015-01-01':'2015-12-31'].std())
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],bins=100,color='green')
sns.distplot(a=returns.loc['2008-01-01':'2008-12-31']['C Return'],bins=100,color='red')

#for loop
for tick in tickers:
    df[tick]['Close'].plot(label=tick)
plt.legend()

#xs
df.xs(key='Close',axis=1,level='Stock Info').plot()

plt.figure(figsize=(12,6))
df['BAC']['Close'].loc['2015-08-01':'2015-09-01'].rolling(window=30).mean().plot(label='30 Day Avg')
df['BAC']['Close'].loc['2015-08-01':'2015-09-01'].plot(label='BAC Close')
plt.legend()
sns.heatmap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True, cmap='coolwarm')
sns.clustermap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
plt.show()
