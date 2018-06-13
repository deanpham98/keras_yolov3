from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import time
import argparse
import functools


import tensorflow as tf
from keras import backend as K
from keras.optimizers import adam
from keras.callbacks import TensorBoard

# Allow relative import
if __name__ == '__main__' and __package__ is None:
       sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
       import yolov3.main
       __package__ = "yolov3.main"


from ..preprocessing.utils.transform import random_transform_generator
from ..preprocessing.utils.enhance import random_enhance_generator
from ..preprocessing.data_generator import DataGenerator
from ..utils.model import yolo_training
from ..utils.callbacks import RedirectModel, Metrics


def parse_args(args):
	parser = argparse.ArgumentParser(description='Training script for YOLOv3 model.')

	parser.add_argument('--training-data', help='Path to training data annotations file.', type=str, required=True)
	parser.add_argument('--validation-data', help='Path to validation data annotaions file.', type=str, required=False, default=None)
	parser.add_argument('--classes-file', help='Path to classes file.', type=str, required=True)
	parser.add_argument('--no-transform', help='Random transformation on image data.', action='store_true')
	parser.add_argument('--no-enhance', help='Random enhancement on image data.', action='store_true')
	parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=10)
	parser.add_argument('--batch-size', help='Minibatch size.', type=int, default=8)
	parser.add_argument('--steps', help='Number of steps per epoch to train.', type=int, default=100)
	parser.add_argument('--tensorboard', help='Logging data infos with tensorboard.', action='store_true')
	parser.add_argument('--weights-path', help='Path to weights file.', type=str, required=True)
	parser.add_argument('--model-path', help='Path to model file.', type=str, required=False, default=None)

	return parser.parse_args(args)

def get_session():

	config = tf.ConfigProto()
	config.log_device_placement = False
	config.gpu_options.allow_growth = True

	return tf.Session(config=config)

def create_generators(args):
	train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations', args.training_data)
	classes_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'classes', args.classes_file)

	assert os.path.exists(train_path), "Training annotaions file does not exist."
	assert os.path.exists(classes_path), "Classes file does not exist."

	transform_generator = None
	enhance_generator = None
	train_generator = None
	validation_generator = None

	if not args.no_transform:
		transform_generator = random_transform_generator(
                    min_rotation=-0.1,
	                max_rotation=0.1,
                    min_translation=(-0.1, -0.1),
                    max_translation=(0.1, 0.1),
                    min_shear=-0.1,
                    max_shear=0.1,
                    min_scaling=(0.9, 0.9),
                    max_scaling=(1.1, 1.1),
                    flip_x_chance=0.5,
                    flip_y_chance=0.0
        )

	if not args.no_enhance:
		enhance_generator = random_enhance_generator(
                brightness_range=(0.2, 1.8),
                contrast_range=(0.5, 1.0),
                sharpness_range=(0.5, 1.5)
            )

	train_generator = DataGenerator(
            data_file           = train_path,
            class_file          = classes_path,
            transform_generator = transform_generator,
            enhance_generator   = enhance_generator,
            batch_size          = args.batch_size,
            training            = True
       )

	if args.validation_data is not None:

		validation_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations', args.validation_data)

		assert os.path.exists(validation_path), "Validation annotations file does not exist."

		validation_generator = DataGenerator(
            data_file  = validation_path,
            class_file = classes_path,
            training   = False
        )

	return train_generator, validation_generator

def create_callbacks(args, infer_model, validation_generator):

	callbacks = []

	tensorboard_callback = None

	if args.tensorboard:
		tensorboard_callback = TensorBoard(
				log_dir                = os.path.join(os.path.dirname(__file__), '..', 'models', 'logs', time.strftime("%d_%m_%Y_%H_%M_%S")),
				batch_size             = args.batch_size,
                write_graph            = True,
                write_grads            = True,
                write_images           = True,
                embeddings_freq        = 0,
                embeddings_layer_names = None,
                embeddings_metadata    = None
       )

		callbacks.append(tensorboard_callback)

	if args.validation_data is not None and validation_generator:
		metrics_callback = Metrics(
				generator   = validation_generator,
		        tensorboard = tensorboard_callback,
			    filepath    = os.path.join('..', 'models', 'weights', '{epoch:03d}-{mAP:.5f}.weights')
        )

		metrics_callback = RedirectModel(metrics_callback, infer_model)

		callbacks.append(metrics_callback)

		return callbacks


def train(args, model, train_generator, callbacks):

        model.fit_generator(
			 generator       = train_generator,
			 steps_per_epoch = args.steps,
			 epochs          = args.epochs,
	         verbose         = 1,
	         callbacks       = callbacks
        )

def main(args=None):
        if args is None:
                args = sys.argv[1:]
        args = parse_args(args)

        sess = get_session()
        K.tensorflow_backend.set_session(sess)

        train_generator, validation_generator = create_generators(args)

        infer_model, model = yolo_training(
		        num_classes  = train_generator.num_classes(),
				model_path   = None, # os.path.join(os.path.dirname(__file__), '..', 'models', '')
				weights_path = args.weights_path
		    )

        print(model.summary())

        callbacks = create_callbacks(
		     	args                 = args,
			    infer_model          = infer_model,
			    validation_generator = validation_generator
	    	)

        model.compile(
		 	loss      = {'yolo_loss': (lambda y_true, y_pred : y_pred)},
		 	optimizer = adam(lr=0.001, decay = 0.00133)
	    )

        train(args, model, train_generator, callbacks)

if __name__ == '__main__':
	main()

