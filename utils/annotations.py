import os
import sys
import pdb


import numpy as np
import cv2

repo_dir = os.path.join(os.path.dirname(__file__), '..')
coco_dir = os.path.join(repo_dir, 'data', 'yicam')
dataset = ('data_cocoformat',)

# Write out annotations
for data in dataset:
        with open(os.path.join(repo_dir, 'data', 'annotations', data + '.txt'), 'a') as file:
                root = os.path.join(coco_dir, data, 'labels')
                listdir = os.listdir(root)
                for filename in listdir:
                        with open(os.path.join(root, filename), 'r') as annot:
                                image_path = os.path.join('yicam', data, 'images', filename)[:-4] + '.jpg'
                                boxes = ''
                                img = cv2.imread(os.path.join(repo_dir, 'data', image_path))
                                size = np.array(img.shape[0:2])
                                for line in annot.readlines():
                                        box = [i.strip() for i in line.split()]
                                        b = np.array([float(i) for i in box[1:]])
                                        b[0:2] = (b[0:2] - b[2:4] / 2)
                                        b[2:4] = b[0:2] + b[2:4]
                                        b[0:2] *= size[::-1]
                                        b[2:4] *= size[::-1]
                                        box[1:] = [str(i) for i in b.astype('int32')]
                                        boxes = boxes + ' ' + ','.join(box[1:]) + ',' + box[0]
                                if boxes != '':
                                        file.write(image_path + boxes + '\n')

