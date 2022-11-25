import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np

encoder = keras.models.load_model('encoder_model')
func = K.function([encoder.get_layer(index=0).input], encoder.get_layer(index=2).output)
save_directory = 'compressed_positions/'
filenames = []
for file in os.listdir('.'):
	if(file.endswith('.csv')):
		filenames.append(file)


for i in range(len(filenames)):
	filename = filenames[i]
	print(i)
	data = pd.read_csv(filename)
	x = np.array(data['Positions'])
	X = []

	for pos in x:
		curr = []
		for num in pos:
			if(num == '1'):
				curr.append(1)
			elif(num == '0'):
				curr.append(0)
		curr = np.array(curr)
		curr = curr.reshape((1,775))
		compressed = func([curr])
		compressed = compressed.reshape((1,128))
		X.append(compressed)


	y = np.array(data['Evaluations'])
	Y = []
	for entry in y:
		Y.append(float(entry))


	d = {'Positions': X, 'Evaluations': Y}
	df = pd.DataFrame(data=d)
	df.to_csv(os.path.join(save_directory, filename), index=False)

