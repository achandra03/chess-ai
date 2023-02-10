import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import activations
import pandas as pd
import numpy as np
import warnings
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
from functools import partial
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
import time
import pprint


def report_perf(optimizer, X, y, title="model", callbacks=None):
	"""
	A wrapper for measuring time and performances of different optmizers

	optimizer = a sklearn or a skopt optimizer
	X = the training set
	y = our target
	title = a string label for the experiment
	"""
	start = time.time()

	if callbacks is not None:
		optimizer.fit(X, y, callback=callbacks)
	else:
		optimizer.fit(X, y)

	d=pd.DataFrame(optimizer.cv_results_)
	best_score = optimizer.best_score_
	best_score_std = d.iloc[optimizer.best_index_].std_test_score
	best_params = optimizer.best_params_

	print('Best parameters:')
	pprint.pprint(best_params)
	print()
	hrs = (time.time() - start) / 3600
	print('took', hrs, 'hours')
	return best_params

warnings.filterwarnings("ignore")
last_filename = ""
train_directory = '/Users/arnavchandra/Desktop/chess/data/positions/train/'
test_directory = '/Users/arnavchandra/Desktop/chess/data/positions/test/'

ma = 10
mi = -10



train_filenames = []
for file in os.listdir(train_directory):
	if(file.endswith('.csv')):
		train_filenames.append(file)

test_filenames = []
for file in os.listdir(test_directory):
	if(file.endswith('.csv')):
		test_filenames.append(file)

random.shuffle(train_filenames)
random.shuffle(test_filenames)

dataset = pd.read_csv('/Users/arnavchandra/Desktop/chess/data/positions/depth1.csv')
X = dataset['Positions']
Y = dataset['Evaluations']
x = []
y = []
uppercnt = 0
lowercnt = 0
middlecnt = 0

for entry in Y:
	val = float(entry)
	converted = val * (ma - mi) + mi
	if(converted >= 2.5):
		uppercnt += 1
	elif(converted <= -2.5):
		lowercnt += 1
	else:
		middlecnt += 1
print(lowercnt, middlecnt, uppercnt)
target = min(min(lowercnt, middlecnt), uppercnt)
lowercnt = 0
middlecnt = 0
uppercnt = 0
madata = -1e30
midata = 1e30
for pos, entry in zip(X, Y):
	val = float(entry)
	converted = val * (ma - mi) + mi
	
	if(converted >= 2.5):
		uppercnt += 1
		if(uppercnt > target):
			continue
	elif(converted <= -2.5):
		lowercnt += 1
		if(lowercnt > target):
			continue
	else:
		middlecnt += 1
		if(middlecnt > target):
			continue
	
	y.append(converted)
	madata = max(madata, converted / ma)
	midata = min(midata, converted / ma)

	curr = []
	currnum = ''
	for i in range(0, len(pos)):
		num = pos[i]
		if(num == ' ' and len(currnum) > 0):
			curr.append(float(currnum))
			currnum = ''
		else:
			if(num != '[' and num != ']' and num != ' '):
				currnum += num
	
	curr.append(float(currnum))	
	curr = np.array(curr)
	curr = curr.reshape((1,65))
	x.append(curr)

x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print('xtrain', x_train.shape)
print('ytrain', y_train.shape)

rows = x_train.shape[0]
cols = x_train.shape[2]
x_train = x_train.reshape((rows, cols))

rows = y_train.shape[0]
y_train = y_train.reshape((rows, 1))

#bst = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8) #0.11 mse
#bst = XGBRegressor(n_estimators=10000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8) #0.07 mse
#bst = XGBRegressor(n_estimators=3251, max_depth=8, learning_rate=0.6793187137881445, colsample_bytree=0.5778677484569327, reg_alpha=85.87438316565517, reg_lambda=98.05562626781331, subsample=0.21526179068277768)

reg = XGBRegressor(random_state=0, booster='gbtree', objective='reg:squarederror')

scoring = make_scorer(partial(MSE, squared=False), greater_is_better=False)

search_spaces = { 'learning_rate': Real(0.01, 1.0, 'uniform'),
                 'max_depth': Integer(2, 14),
                 'subsample': Real(0.1, 1.0, 'uniform'),
                 'colsample_bytree': Real(0.1, 1.0, 'uniform'), # subsample ratio of columns by tree
                 'reg_lambda': Real(1e-9, 100., 'uniform'), # L2 regularization
                 'reg_alpha': Real(1e-9, 100., 'uniform'), # L1 regularization
                 'n_estimators': Integer(50, 50000)
}

opt = BayesSearchCV(estimator=reg,
                    search_spaces=search_spaces,
                    scoring=scoring,
                    n_iter=120,                                       # max number of trials
                    n_points=1,                                       # number of hyperparameter sets evaluated at the same time
                    n_jobs=1,                                         # number of jobs
                    iid=False,                                        # if not iid it optimizes on the cv score
                    return_train_score=False,
                    refit=False,
                    optimizer_kwargs={'base_estimator': 'GP'},        # optmizer parameters: we use Gaussian Process (GP)
                    random_state=0)                                   # random state for replicability


overdone_control = DeltaYStopper(delta=0.0001)   
time_limit_control = DeadlineStopper(total_time=60*60*4)
print('starting')
best_params = report_perf(opt, x_train, y_train,'XGBoost_regression', callbacks=[overdone_control, time_limit_control])
'''
bst.fit(x_train, y_train)
rows = x_test.shape[0]
cols = x_test.shape[2]
x_test = x_test.reshape((rows, cols))

rows = y_test.shape[0]
y_test = y_test.reshape((rows, 1))

pred = bst.predict(x_test)
mse = MSE(pred, y_test)
print('mse is', mse)
bst.save_model('boost_uncompressed_bayesian.json')
'''
