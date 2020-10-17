import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
df = pd.read_csv('train.csv')
sns.jointplot(x='Fare',y='Age',data=df)
plt.show()