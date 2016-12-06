import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import exp

import statsmodels.api as sm


def plot_chapter_31():
	groups = data.groupby('f')

	# Plot
	fig, ax = plt.subplots()
	# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
	    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
	ax.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('chapter3-1.png')

def plot_chapter_32():
	X = np.linspace(-4, 4, 50)
	Bs = [[-2, -0.8], [-1, 0.4]]
	fig, ax = plt.subplots()
	for bs in Bs:
		Y = [exp(bs[0] + bs[1]*x) for x in X]
		ax.plot(X, Y, label='{b1=%.2f, b2=%.2f}' % (bs[0], bs[1]))
	ax.legend()
	plt.xlabel('x_i')
	plt.ylabel('lambda_i')
	plt.savefig('chapter3-2.png')

def plot_chapter_33():
	xs = pd.DataFrame({'x': np.linspace(5, 20, 50)})
	fig, axes = plt.subplots(2)
	def log_link_func(x, d):
		return exp(1.26 + 0.08 * x - 0.032 * d)
	def iden_link_func(x, d):
		return 1.26 + 0.08 * x - 0.032 * d
	for d in [0, 1]:
		xs['lambda_log'] = xs['x'].apply(log_link_func, args=(d,))
		xs['lambda_iden'] = xs['x'].apply(iden_link_func, args=(d,))
		axes[0].plot(xs['x'], xs['lambda_log'], label='use log with d=%d' % d)
		axes[1].plot(xs['x'], xs['lambda_iden'], label='use identity with d=%d' % d)
	axes[0].legend(loc='upper center')
	axes[1].legend(loc='upper center')
	plt.savefig('chapter3-3.png')

def glm():
	pass

if __name__ == '__main__':
	data = pd.read_csv('http://hosho.ees.hokudai.ac.jp/~kubo/stat/iwanamibook/fig/poisson/data3a.csv')

	# print data
	# print data.describe()

	# plot_chapter_31()	
	# plot_chapter_32()	

	# X = sm.add_constant(data['x'], prepend=False)
	# glm_posi = sm.GLM(data['y'], X, family=sm.families.Poisson())
	# res = glm_posi.fit()
	# print res.summary()

	# F = sm.add_constant(data['f'], prepend=False)
	# F['f'] = F['f'].apply(lambda x: 0 if x == 'C' else 1)
	# glm_posi = sm.GLM(data['y'], F, family=sm.families.Poisson())
	# res = glm_posi.fit()
	# print res.summary()

	# XF = sm.add_constant(data[['x','f']])
	# XF['f'] = XF['f'].apply(lambda x: 0 if x == 'C' else 1)
	# glm_posi = sm.GLM(data['y'], XF, family=sm.families.Poisson())
	# res = glm_posi.fit()
	# print res.summary()	

	plot_chapter_33()