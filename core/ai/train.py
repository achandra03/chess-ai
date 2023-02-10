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



warnings.filterwarnings("ignore")
last_filename = ""
train_directory = '/Users/arnavchandra/Desktop/chess/data/positions/train/'
test_directory = '/Users/arnavchandra/Desktop/chess/data/positions/test/'

ma = 10
mi = -10


def generate_batches(file_list, batch_size, file_directory):
	cnt = 0
	while True:
		file = file_list[cnt]
		last_filename = file
		cnt = (cnt + 1) % len(file_list)
		data = pd.read_csv(file_directory + file)

		x = np.array(data['Positions'])
		y = np.array(data['Evaluations'])


		X = []
		Y = []

		for pos, entry in zip(x, y):
			val = float(entry)
			converted = val * (ma - mi) + mi
			if(converted >= -2.5):
				continue

			Y.append(val)

			curr = []
			currnum = ''
			for i in range(0, len(pos)):
				num = pos[i]
				if(num == ' ' and len(currnum) > 0):
					curr.append(float(currnum))
					currnum = ''
				else:
					if(num != '[' and num != ']'):
						currnum += num
			curr.append(float(currnum))	
			curr = np.array(curr)
			curr = curr.reshape((1,64))
			X.append(curr)

		X = np.array(X)
		Y = np.array(Y)
		for idx in range(0, X.shape[0], batch_size):
			X_loc = X[idx:(idx + batch_size)]
			Y_loc = Y[idx:(idx + batch_size)]

			
			yield X_loc, Y_loc



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
	
	y.append(converted / ma)
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


train_generator = generate_batches(train_filenames, 128, train_directory)
test_generator = generate_batches(test_filenames, 128, test_directory)

initializer = tf.keras.initializers.GlorotNormal()
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(2048, input_shape=(None, 65), kernel_initializer=initializer, activation='relu'))
model.add(tf.keras.layers.Dense(1024, kernel_initializer=initializer, activation='relu'))
model.add(tf.keras.layers.Dense(512, kernel_initializer=initializer, activation='relu'))
model.add(tf.keras.layers.Dense(256, kernel_initializer=initializer, activation='relu'))
model.add(tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu'))
model.add(tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu'))
model.add(tf.keras.layers.Dense(1, kernel_initializer=initializer, activation='tanh'))


#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')

model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
model.save('model')
