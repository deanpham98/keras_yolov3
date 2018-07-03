import os
import sys
import pdb
import argparse

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.initializers import glorot_uniform

# Allow relative import
if __name__ == '__main__' and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import yolov3.utils
	__package__ = 'yolov3.utils'

from .model import yolo_training

def get_classes(filepath):
	classes = None
	with open(filepath, 'r') as file:
		classes = [c.strip('\n') for c in file.readlines()]
	return classes

def get_coco_classes(coco_filepath):
	coco_classes = None
	with open(coco_filepath, 'r') as file:
		coco_classes = {key.strip('\n'): value for value, key in enumerate(file.readlines())}
	return coco_classes

def write_new_classes_file(filepath, new_classes):
	with open(filepath, 'w') as file:
		for line in new_classes:
			file.write(line + '\n')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Extract useful weights for fast convergent training.')
	parser.add_argument('--filename', help='Name of file containing desired classes.', type=str, required=True)
	parser.add_argument('--new-model', help='Name of file containing new model.', type=str, required=True)
	parser.add_argument('--new-weights', help='Name of file containing new weights.', type=str, required=True)
	flags = parser.parse_args()

	coco_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'classes', 'coco_classes.txt')
	yolov3_original_model_path = os.path.join(os.path.dirname(__file__), '..', 'model_data', 'models', 'original.h5')

	g = tf.Graph()

	with g.as_default():
		initializer = glorot_uniform(seed=10101)

	tf.reset_default_graph()

	model = load_model(yolov3_original_model_path, compile=False)

	coco_classes = get_coco_classes(coco_filepath)

	filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'classes', flags.filename)

	assert filepath[-4:] == '.txt' and os.path.exists(filepath)

	classes = get_classes(filepath)

	new_model, _ = yolo_training(
		num_classes  = len(classes),
		weights_path = 'original.weights',
		freeze_body  = 'all'
	)

	indices = [coco_classes[c] + 5 for c in classes if c in coco_classes]

	new_classes = [c for c in classes if c in coco_classes] + [c for c in classes if c not in coco_classes]

	for i in range(-3, 0):

		weights = model.layers[i].get_weights()[0]
		bias = model.layers[i].get_weights()[1]
		new_weights = np.zeros(weights.shape[:-1] + (3*(5+len(classes)),))
		new_bias = np.zeros((3*(5+len(classes)),))
		for j in range(3):
			new_weights[..., j*(5+len(classes)): j*(5+len(classes)) + 5] = weights[..., j*(5 + len(coco_classes)): j*(5 + len(coco_classes)) + 5]
			new_bias[j*(5+len(classes)): j*(5+len(classes)) + 5] = bias[j*(5 + len(coco_classes)): j*(5 + len(coco_classes)) + 5]
			for k, m in enumerate(indices):
				new_weights[..., j*(5+len(classes)) + k+5] = weights[..., j*(5+len(coco_classes)) + m]
				new_bias[j*(5+len(classes)) + k+5] = bias[j*(5+len(coco_classes)) + m]

			with tf.Session(graph=g) as sess:
				new_weights[..., j*(5+len(classes)) + len(indices)+5: j*(5+len(classes)) + len(classes)+5] = initializer.__call__(shape=new_weights[..., len(indices):len(classes)].shape).eval()
				new_bias[j*(5+len(classes)) + len(indices)+5: j*(5+len(classes)) + len(classes)+5] = initializer.__call__(shape=new_bias[len(indices):len(classes)].shape).eval()

		new_model.layers[i].set_weights([new_weights, new_bias])

	new_model.save(os.path.join(os.path.dirname(__file__), '..', 'model_data', 'models', flags.new_model))
	new_model.save_weights(os.path.join(os.path.dirname(__file__), '..', 'model_data', 'weights', flags.new_weights))
	write_new_classes_file(filepath, new_classes)
