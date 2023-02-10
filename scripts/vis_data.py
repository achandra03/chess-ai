import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import smogn

filename = '/Users/arnavchandra/Desktop/chess/data/chessData.csv'

d = []

def process_evaluation(evaluation):
	x = 0
	ma = 10
	mi = -10
	if(evaluation[0] == '#'):
		if(evaluation[1] == '+'):
			x = ma
		else:
			x = mi
	elif(evaluation[0] == '+'):
		x = int(evaluation[1:])
		x = x / 100
		x = min(x, ma)
	elif(evaluation[0] == '-'):
		x = -1 * int(evaluation[1:])
		x = x / 100
		x = max(x, mi)
	return x

data = pd.read_csv(filename)
'''
evals = np.array(data['Evaluation'])
new_evals = []
for entry in evals:
	new_evals.append(process_evaluation(entry))

data = data.drop('Evaluation', axis=1)
data = data.drop('FEN', axis=1)
data['Evaluation'] = new_evals
balanced = smogn.smoter(data=data, y='Evaluation')
print(type(balanced))
exit(0)
'''
y = np.array(data['Evaluation'])
cnt = 0
for entry in y:
	val = process_evaluation(entry)
	if(val < -2.5 or val > 2.5):
		d.append(val)
	cnt += 1
	print(cnt)

plt.hist(d)
plt.show()
