import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
# read train.csv file
raw = pd.read_csv("train.csv")
raw.drop(raw.columns[[0, 1]], axis=1, inplace=True)
raw = raw.as_matrix()
raw = np.ravel(raw)
# define x , y
x = []
y = []

for m in range(12):
	for d in range(234):
		x.append(raw[m*12+d:m*12+d+5])
		y.append(raw[m*12+d+5])

x = np.array(x)
y = np.array(y)

# train using linear regression
l_rate = 0.000000001   # define learning rate
repeat = 10000 # define repeat time
###### START TODO's ######
# here you need to update (w1, w2, w3, w4, w5) according to their gradient
# step 1. init (w1, w2, w3, w4, w5)

for i in range(repeat):
	# step 2. calculate loss
	loss = 0
	# print cost every iteration
	cost = np.sum(loss**2) / len(x)
	cost = np.sqrt(cost)
	print('iteration: %d | cost: %f' %(i, cost))
###### END TODO's ######

# after you finish TODO's un-comment this part
# visualizing your work
'''
predicted = np.dot(x,np.transpose([w1,w2,w3,w4,w5]))
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''