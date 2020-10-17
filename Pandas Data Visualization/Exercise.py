import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('df3')
import seaborn as sns
#Q1
'''
** Recreate this scatter plot of b vs a. Note the color and size of the points. 
Also note the figure size. See if you can figure out how to stretch it in a similar fashion. 
Remeber back to your matplotlib lecture...**
'''
# df.plot.scatter(x='a',y='b',c='red',s=50)

#Q2
# ** Create a histogram of the 'a' column.**
# df['a'].plot.hist()

#Q3
#** These plots are okay, but they don't look very polished
# . Use style sheets to set the style to 'ggplot' and redo
#  the histogram from above. Also figure out 
# how to add more bins to it.***
# plt.style.use('ggplot')
# df['a'].hist()

#Q4
# ** Create a boxplot comparing the a and b columns.**
# df[['a','b']].plot.box()

#Q5
# ** Create a kde plot of the 'd' column **
# df['d'].plot.kde()

#Q5
# ** Figure out how to increase the linewidth and make the
#  linestyle dashed. (Note: You would usually not dash a 
# kde plot line)**
# df['d'].plot.kde(ls='--',lw=6,color='red')

#Q6
# ** Create an area plot of all the columns for 
# just the rows up to 30. (hint: use .ix).**
# Dataframe doesn't have ix attribute, so i'm using iloc attribute
# df.iloc[0:31].plot.area()

#Q7
'''
Note, you may find this really hard, reference the 
solutions if you can't figure it out!
** Notice how the legend in our previous figure overlapped
some of actual diagram. Can you figure out how to display
the legend outside of the plot as shown below?**
'''
df.iloc[0:31].plot.area()
plt.legend(loc='center left',bbox_to_anchor=(1.0,.5))
plt.show()