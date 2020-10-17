import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
titanic = pd.read_csv('titanic.csv')
#Question 1
#sns.jointplot(x='fare',y='age',data=titanic)
#Question 2
# a = sns.distplot(titanic['fare'],kde=False,color='salmon')
# a.set(xlim=(0,600))
# a.set(ylim=(0,500))
#Question 3
print(titanic.info())
plt.show()