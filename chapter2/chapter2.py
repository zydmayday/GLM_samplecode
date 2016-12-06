import pandas as pd
import math
import matplotlib.pyplot as plt

def poisson_probability(actual, mean):
    # naive:   math.exp(-mean) * mean**actual / factorial(actual)

    # iterative, to keep the components from getting too large or small:
    p = math.exp(-mean)
    for i in xrange(actual):
        p *= mean
        p /= i+1
    return p

def hist(data):
	ax = data.hist(bins=8) 
	fig = ax.get_figure()
	return ax, fig
	# fig.savefig('/Users/zhangyidong/Documents/git/glm/chapter2-1.png')

def plot(data, ax=None):
	if ax:
		ax = data2.plot(style= 'ro-', ax=ax)
	else:
		ax = data2.plot(style= 'ro-')
	fig = ax.get_figure()
	return ax, fig

if __name__ == '__main__':
	data = pd.Series([2,2,4,6,4,5,2,3,1,2,0,4,3,3,3,3,4,2,7,2,4,3,3,3,4,3,7,5,3,1,7,6,4,6,5,2,4,7,2,2,6,2,4,5,4,5,1,3,2,3])
	print len(data)
	print data.describe()
	print data.count()

	# ax = data.hist(bins=8) 
	# fig = ax.get_figure()
	# fig.savefig('/Users/zhangyidong/Documents/git/glm/chapter2-1.png')
	# ax, fig = hist(data)

	X, Y = [], []

	print '%s %5s' % ('y', 'prob')
	for i in range(10):
		X.append(i)
		prob = poisson_probability(i, 3.56)
		Y.append(prob)
		print '%s %5f' % (i, prob)

	data2 = pd.DataFrame([y * 50 for y in Y], X)
	# ax, fig = plot(data2)
	# ax = data2.plot(style= 'ro-', ax=ax)
	# fig = ax.get_figure()
	# fig.savefig('/Users/zhangyidong/Documents/git/glm/chapter2-3.png')

	# plt.plot(X, Y, 'ro-')
	# plt.ylabel('prob')
	# plt.xlabel('y')
	# plt.savefig('/Users/zhangyidong/Documents/git/glm/chapter2-2.png')

