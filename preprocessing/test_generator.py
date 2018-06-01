import cv2
from data_generator import DataGenerator
from utils.transform import random_transform_generator
from utils.enhance import random_enhance_generator


data_file = '/models/2007_train.txt'
class_file = '/models/model_data/voc_classes.txt'
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
                          group_method=group_method)

for i in range(5):
	next_batch = train_gen.next()
	images = next_batch[0]
	for image in images:
		cv2.imshow("image", image)
		while True:
			if cv2.waitKey(0):
				cv2.destroyWindow("image")
				break




