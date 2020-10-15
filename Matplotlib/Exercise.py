import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,100)
y = x*2
z = x**2
#** Import matplotlib.pyplot as plt and set %matplotlib inline if you 
# are using the jupyter notebook. What command do you use if you aren't using the jupyter notebook?**
'''
** Follow along with these steps: **
* ** Create a figure object called fig using plt.figure() **
* ** Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax. **
* ** Plot (x,y) on that axes and set the labels and titles to match the plot below:**
'''
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('title')
plt.show()
'''
** Create a figure object and put two axes on it, 
ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] 
respectively.**
'''
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.2,.2])
