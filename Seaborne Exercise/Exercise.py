import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
titanic = pd.read_csv('titanic.csv')
#Q1
# sns.jointplot(x='fare',y='age',data=titanic)

#Q2
# graph = sns.distplot(titanic['fare'],kde=False,color='red')
# graph.set(xlim=(0,600),ylim=(0,500))

#Q3
# sns.boxplot(x='class',y='age',data=titanic)

#Q4
# sns.swarmplot(x='class',y='age',data=titanic)

#Q5
# sns.countplot(x='sex',data=titanic)

#Q6
# sns.heatmap(data=titanic.corr(),cmap='coolwarm')

#Q7
sns.FacetGrid(titanic, col='sex').map(plt.hist,'age')
plt.show()