import os
import sys
import pdb

import numpy as np
from PIL import Image

# Allow relative import
if __name__ == '__main__' and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import yolov3.utils
	__package__ = 'yolov3.utils'

# Initialize
repo_dir = os.path.join(os.path.dirname(__file__), '..')
coco_dir = os.path.join(repo_dir, 'data', 'COCO')
dataset = ('train2014', 'val2014')

# Load classes
coco_classes = None
with open(os.path.join(repo_dir, 'data', 'classes', 'coco_classes.txt'), "r") as file:
	coco_classes = {str(key): value.strip("\n") for key, value in enumerate(file.readlines())}

classes = None
with open(os.path.join(repo_dir, 'data', 'classes', 'classes.txt'), "r") as file:
	classes = {key.strip("\n"): str(value) for value, key in enumerate(file.readlines())}

print('classes:', classes)
# Write out annotations
for data in dataset:
	with open(os.path.join(repo_dir, 'data', 'annotations', data + '.txt'), 'a') as file:
		root = os.path.join(coco_dir, data, 'labels')
		listdir = os.listdir(root)
		for k, filename in enumerate(listdir):
			print('\r{}/{}'.format(k, len(listdir)), end='')
			if 'COCO_train2014_000000167126' in filename:
				continue
			with open(os.path.join(root, filename), 'r') as annot:
				image_path = os.path.join('COCO', data, 'images', filename)[:-4] + '.jpg'
				image = Image.open(os.path.join(os.path.dirname(__file__), '..', 'data', image_path))
				width, height = image.size
				boxes = ''
				for line in annot.readlines():
					box = [i.strip() for i in line.split()]
					if coco_classes[box[0]] not in classes:
						continue
					b = np.array([float(i) for i in box[1:]])
					b[0] *= width
					b[1] *= height
					b[2] *= width
					b[3] *= height
					b[0:2] = b[0:2] - b[2:4] / 2
					b[2:4] = b[0:2] + b[2:4]
					update = ' ' + ','.join([str(int(coord)) for coord in b]) + ',' + classes[coco_classes[box[0]]]
					if update not in boxes:
						boxes = boxes + update
				if boxes != '':
					file.write(image_path + boxes + '\n')
