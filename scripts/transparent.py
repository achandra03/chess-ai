import cv2
import numpy as np
import os

filedir = '/Users/arnavchandra/Desktop/chess/core/game/assets/pieces/'

def convert(filename):
	# load image
	img = cv2.imread(filedir + filename)

	height, width, _ = img.shape

	for i in range(height):
		for j in range(width):
			if img[i, j].sum() == 0:
				img[i, j] = [255, 255, 255]	

	cv2.imwrite(filename, img)

for file in os.listdir(filedir):
	if(not file.endswith('.png')):
		continue
	if(file[0] == 'w'):
		continue
	convert(file)
	
