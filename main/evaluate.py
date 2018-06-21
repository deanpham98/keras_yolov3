import os
import sys
import argparse

# Allow relative import
if __name__ == '__main__' and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import yolov3.main
	__package__ = 'yolov3.main'

from ..utils.model import yolo_training
from ..utils.metrics import evaluate
from ..preprocessing.data_generator import DataGenerator


def parse_args(args):
	parser = argparse.ArgumentParser(description='Evaluation script for YOLOv3 model.')
	parser.add_argument('--evaluation-data', help='Path to evaluation data annotations file.', type=str, required=True)
	parser.add_argument('--classes-file', help='Path to classes file.', type=str, required=True)
	parser.add_argument('--weights-path', help='Path to weights file.', type=str, required=True)
	parser.add_argument('--model-path', help='Path to model file.', type=str, required=False, default=None)

	return parser.parse_args(args)

def create_generator(args):

	evaluation_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations', args.evaluation_data)
	classes_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'classes', args.classes_file)
	assert os.path.exists(evaluation_path), "Evaluation annotations file does not exist."
	assert os.path.exists(classes_path), "Classes file does not exist."

	evaluation_generator = DataGenerator(
		data_file  = evaluation_path,
		class_file = classes_path,
		training   = False
	)

	return evaluation_generator

def create_model(args, generator):

	assert args.weights_path is not None, "Weights path must be provided"

	infer_model, model = yolo_training(
		num_classes  = generator.num_classes(),
		model_path   = args.model_path,
		weights_path = args.weights_path,
		freeze_body  = 'all'
	)

	print(infer_model.summary())

	return infer_model

def main(args=None):

	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	evaluation_generator = create_generator(args)

	model = create_model(args, evaluation_generator)

	APs = evaluate(
		generator       = evaluation_generator,
		model           = model,
		iou_threshold   = 0.5,
		score_threshold = 0.3
	)

	print('mAP:', sum(APs.values()) / len(APs), end='\n')

	for label, AP in APs.items():
		print(evaluation_generator.label_to_name(label), '{:.4f}'.format(AP))


if __name__ == '__main__':
	main()
