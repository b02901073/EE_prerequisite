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
		####################################### START TODO's #######################################
		# In this part of code, please deside how many hours of data to use
		# default = 5 hours
		hour_prior = 5
		# x is an array of pm2.5 of [hour1 hour2 hour3 hour4 hour5 bias]
		tmp = raw[m*12 + d : m*12 + d + hour_prior]
		tmp = np.append(tmp, [1])
		x.append(tmp)
		# y is value of pm2.5 of [hour6]
		y.append(raw[m*12 + d + hour_prior])
		####################################### END TODO's #######################################

x = np.array(x)
y = np.array(y)

# train pm2.5 using linear regression
####################################### START TODO's #######################################
# define learning rate
# small learning rate trains slower but steadily
l_rate = 0.000000001

# define repeat time
# large repeat time may get closer to the minimun of loss
repeat = 100

# here you need to update (w1, w2, w3, w4, w5, b) according to their gradient
# step 1. init (w1, w2, w3, w4, w5, b) 
w1 = 0
w2 = 0
w3 = 0
w4 = 0
w5 = 0
b  = 0
for i in range(repeat):
	# step 2. calculate loss
	y_pred = np.dot(x,np.transpose([w1,w2,w3,w4,w5,b]))
	loss = y_pred - y
	gradient = np.dot(np.transpose(x),loss)
	[w1,w2,w3,w4,w5,b] = [w1,w2,w3,w4,w5,b] - l_rate * gradient
	
	# print cost every iteration
	cost = np.sum(loss**2) / len(x)
	cost = np.sqrt(cost)
	print('iteration: %d | cost: %f' %(i, cost))
####################################### END TODO's #######################################

# let's see what you have trained
print('')
print('after training for %d times, you get' %(repeat))
print('w1 = ', w1)
print('w2 = ', w2)
print('w3 = ', w3)
print('w4 = ', w4)
print('w5 = ', w5)
print('b  = ', b)

# un-comment this part of code after you trained
# you can see how close your predition and correct answer is
'''
predicted = np.dot(x,np.transpose([w1,w2,w3,w4,w5,b]))
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''