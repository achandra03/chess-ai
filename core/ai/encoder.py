import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import random
import pandas as pd

train_directory = '/Users/arnavchandra/Desktop/chess/data/positions/train/'
test_directory = '/Users/arnavchandra/Desktop/chess/data/positions/test/'

def generate_batches(file_list, batch_size, file_directory):
	cnt = 0
	while True:
		file = file_list[cnt]
		last_filename = file
		cnt = (cnt + 1) % len(file_list)
		data = pd.read_csv(file_directory + file)

		x = np.array(data['Positions'])


		X = []


		for pos in x:
			curr = []
			for num in pos.split():
				if(num[0] == '['):
					num = num[1:]
				if(num[len(num) - 1] == ']'):
					num = num.rstrip(num[-1])

				if(num == 'True'):
					curr.append(1)
				elif(num == 'False'):
					curr.append(0)
				else:
					curr.append(int(num))

			X.append(curr)

		X = np.array(X)


		for idx in range(0, X.shape[0], batch_size):
			X_loc = X[idx:(idx + batch_size)]
			
			yield X_loc, X_loc




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


'''
encoder = tf.keras.Sequential()
encoder.add(tf.keras.layers.Dense(512, input_shape=(775,), activation='relu'))
encoder.add(tf.keras.layers.Dense(256, activation='relu'))
encoder.add(tf.keras.layers.Dense(128, activation='relu', name='encoded_output'))
encoder.add(tf.keras.layers.Dense(256, activation='relu'))
encoder.add(tf.keras.layers.Dense(512, activation='relu'))
encoder.add(tf.keras.layers.Dense(775, activation='sigmoid', name='decoded_output'))

save = tf.keras.callbacks.ModelCheckpoint(filepath='encoder_weights.h5', save_weights_only=True, save_best_only=True)

encoder.compile(optimizer='adam', loss='binary_crossentropy')
encoder.load_weights('encoder_weights.h5')
#encoder.save('encoder_model')

encoder.fit(steps_per_epoch=len(train_filenames), workers=1, x=train_generator, max_queue_size=32, epochs=100, callbacks=[save], validation_data=test_generator, validation_steps=len(test_filenames), batch_size=256)
'''
model = keras.models.load_model('static_evaluation_model')
model.fit(steps_per_epoch=len(train_filenames), workers=1, x=train_generator, max_queue_size=32, epochs=100, validation_data=test_generator, validation_steps=len(test_filenames), batch_size=256)

