import os
import sys
import json
import pdb


import cv2

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
annotations_dir = os.path.join(data_dir, 'annotations')
bdd_dir = os.path.join(data_dir, 'BDD')
images_dir = os.path.join(bdd_dir, 'images', '100k')
labels_dir = os.path.join(bdd_dir, 'labels', '100k')
data = ['train', 'val']

classes = {
	'bus': 0,
	'car': 1,
	'person': 2,
	'rider': 3,
	'traffic light': 4,
	'traffic sign': 5,
	'truck': 6
}


def extract_labels(file, annot, dataset):
	frame = file['frames'][0]
	image_name = 'BDD/images/100k/' + dataset + '/' + file['name'] + '.jpg'
	image = cv2.imread(os.path.join(os.path.dirname(__file__), '..', 'data', image_name))
	bbox = ''
	for obj in frame['objects']:
		if 'box2d' not in obj or obj['category'] not in classes:
			continue
		coords = obj['box2d']
		if coords['x1'] >= coords['x2'] or coords['y1'] >= coords['y2']:
			continue
		bbox += ' {},{},{},{},{}'.format(int(coords['x1']), int(coords['y1']), int(coords['x2']), int(coords['y2']), classes[obj['category']])
	if bbox != '':
		annot.write(image_name + bbox + '\n')


for dataset in data:
	with open(os.path.join(annotations_dir, 'bdd_{}.txt'.format(dataset)), 'a') as annot:
		for file in os.listdir(os.path.join(labels_dir, dataset)):
			if file[-5:] == '.json':
				file = os.path.join(labels_dir, dataset, file)
				extract_labels(json.load(open(file, 'r')), annot, dataset)

