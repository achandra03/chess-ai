import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


data1 = pd.read_csv('/Users/arnavchandra/Desktop/chess/data/positions/1.csv')
data2 = pd.read_csv('/Users/arnavchandra/Desktop/chess/data/positions/2.csv')

data = data1.append(data2, ignore_index=False)

x = np.array(data['Positions'])
Y = np.array(data['Evaluations'])

X = []

for pos in x:
	curr = []
	for num in pos:
		if(num == '-1'):
			curr.append(-1)
		elif(num == '1'):
			curr.append(1)
		elif(num == '0'):
			curr.append(0)
	X.append(curr)

X = np.array(X)



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2048, input_shape=(768,), activation='elu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2048, activation='elu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2048, activation='elu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


opt = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.7, nesterov=True)
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
model.compile(optimizer=opt, loss='mse')
model.fit(X, Y, epochs=100000, callbacks=[stop], validation_split=0.2)

model.save_weights('weights.h5')
