import os
import sys

import cv2

if __name__ == '__main__' and __package__ is None:
      sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
      import yolov3.preprocessing
      __package__ = 'yolov3.preprocessing'
 
from .data_generator import DataGenerator
from .utils.transform import random_transform_generator
from .utils.enhance import random_enhance_generator


data_file = '../data/annotations/2007_train.txt'
class_file = '../data/classes/voc_classes.txt'
transform_generator = random_transform_generator(
              min_rotation=-0.1, \
              max_rotation=0.1, \
              min_translation=(-0.1, -0.1), \
              max_translation=(0.1, 0.1), \
              min_shear=-0.1, \
              max_shear=0.1, \
              min_scaling=(0.9, 0.9), \
              max_scaling=(1.1, 1.1), \
              flip_x_chance=0.5, \
              flip_y_chance=0, \
)
enhance_generator = random_enhance_generator()
batch_size = 16
group_method = 'random'

train_gen = DataGenerator(data_file=data_file, \
                          class_file=class_file, \
                          transform_generator=transform_generator, \
                          enhance_generator=enhance_generator, \
                          batch_size=batch_size, \
                          group_method=group_method, \
                          training=True)

for i in range(5*train_gen.size()):
  print('{} out of {}'.format(i, 5*train_gen.size()))
  next_batch = train_gen.next()





