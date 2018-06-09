import os
import sys
# import pdb

# Allow relative import
if __name__ == '__main__' and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import yolov3.utils
	__package__ = 'yolov3.utils'

# Initialize 
repo_dir = os.path.join(os.path.dirname(__file__), '..')
coco_dir = os.path.join(repo_dir, 'data', 'COCO')
dataset = ('train2014', 'val2014')

# pdb.set_trace()

# Load classes
coco_classes = None
with open(os.path.join(repo_dir, 'data', 'classes', 'coco_classes.txt'), "r") as file:
	coco_classes = {str(key): value.strip("\n") for key, value in enumerate(file.readlines())}

classes = None
with open(os.path.join(repo_dir, 'data', 'classes', 'classes.txt'), "r") as file:
	classes = {key.strip("\n"): str(value) for value, key in enumerate(file.readlines())}

# Write out annotations
for data in dataset:
	with open(os.path.join(repo_dir, 'data', 'annotations', data + '.txt'), 'a') as file:
		root = os.path.join(coco_dir, data, 'annotations')
		listdir = os.listdir(root)
		for filename in listdir:
			with open(os.path.join(root, filename), 'r') as annot:
				image_path = os.path.join('COCO', data, 'images', filename)[:-4] + '.jpg'
				boxes = ''
				for line in annot.readlines():
					box = [i.strip() for i in line.split()]
					if coco_classes[box[0]] not in classes:
						continue
					boxes = boxes + ' ' + ','.join(box[1:]) + ',' + classes[coco_classes[box[0]]]
				if boxes != '':
					file.write(image_path + boxes + '\n')
