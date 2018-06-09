import os
import sys


import numpy as np
import cv2

dataset = ("train2014.txt", "val2014.txt")
data_dir = os.path.join(ps.path.dirname(__file__), '..', 'data')
h = [True, False, True, False]
w = [False, True, False, True]
for dset in dataset:
	lines = None
	with open(os.path.join(data_dir, 'annotations', dset), "r") as file:
		lines = [line.strip('\n').split(' ') for line in file.readlines()]
	with open(os.path.join(data_dir, 'annotations', 'coco_' + dset), 'a') as file:
		for line in lines:
			img = cv2.imread(os.path.join(data_dir, line[0]))
			height, width, channel = img.shape
			data = line[0]
			for box in line[1:]:
				box = box.split(',')
				box_to_float = np.array([float(i) for i in box[:4]])
				box_to_float[h] *= width
				box_to_float[w] *= height
				box_to_float[0:2] = box_to_float[0:2] - box_to_float[2:4] / 2
				box_to_float[2:4] = box_to_float[0:2] + box_to_float[2:4]
				data = data + ' ' + ','.join([str(i) for i in box_to_float.astype('int32').tolist()] + [box[4]])
			file.write(data + '\n')


