import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import warnings
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

warnings.filterwarnings("ignore")
last_filename = ""
train_directory = '/Users/arnavchandra/Desktop/chess/data/positions/compressed_train/'
test_directory = '/Users/arnavchandra/Desktop/chess/data/positions/compressed_test/'
encoder = keras.models.load_model('encoder_model')

func = K.function([encoder.get_layer(index=0).input], encoder.get_layer(index=2).output)


def inv_sigmoid(x):
	return np.log(x / (1 - x))

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

		for entry in y:
			Y.append(float(entry))

		Y = np.array(Y)

		for pos in x:
			curr = []
			currstr = ''
			for num in pos:
				if(num == '[' or num == ']'):
					continue
				if(num == ' ' and currstr == ''):
					continue
				elif(num == ' ' and currstr != ''):
					curr.append(float(currstr))
					currstr = ''
				else:
					currstr += num
			if(currstr != ''):
				curr.append(float(currstr))

			curr = np.array(curr)
			curr = curr.reshape((1,128))
			X.append(curr)

		X = np.array(X)

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

train_generator = generate_batches(train_filenames, 10000, train_directory)
test_generator = generate_batches(test_filenames, 10000, test_directory)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2048, input_shape=(128,), activation='relu'))
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2048, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2048, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1, activation='tanh'))


stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
save = tf.keras.callbacks.ModelCheckpoint(filepath='weights.h5', save_weights_only=True, save_best_only=True)
model.compile(optimizer='adam', loss='mse')
model.save("my_model_v2")
model.fit(steps_per_epoch=len(train_filenames), workers=1, x=train_generator, max_queue_size=32, epochs=100000, callbacks=[stop, save], validation_data=test_generator, validation_steps=len(test_filenames), batch_size=128)
print(last_filename)
