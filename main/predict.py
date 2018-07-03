#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import sys
import random
import argparse
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

if __name__ == '__main__':
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import yolov3.main
	__package__ = 'yolov3.main'

from ..utils.model import yolo_eval, yolo_training
from ..utils.pascal_voc_io import PascalVocWriter
from ..preprocessing.utils.image import letterbox_image


def parse_args(args):
	parser = argparse.ArgumentParser(description='Script for image or video object detection.')
	parser.add_argument('--weights-file', help='Name of weights file.', type=str, required=True)
	parser.add_argument('--anchors-file', help='Name of anchors file.', type=str, required=True)
	parser.add_argument('--classes-file', help='Name of classes file.', type=str, required=True)
	parser.add_argument('--score', help='Score threshold for filtering.', type=float, required=False, default=0.3)
	parser.add_argument('--iou', help='IOU threshold for non-max suppression.', type=float, required=False, default=0.5)
	parser.add_argument('--generate-data', help='Flag that indicates generating data.', action='store_true')
	return parser.parse_args()


class YOLO(object):
	def __init__(self, args):
		self.weights_path = args.weights_file
		self.anchors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'anchors', args.anchors_file)
		self.classes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'classes', args.classes_file)
		self.score = args.score
		self.iou = args.iou
		self.class_names = self._get_class()
		self.anchors = self._get_anchors()
		self.sess = K.get_session()
		self.model_image_size = (416, 416) # fixed size or (None, None)
		self.is_fixed_size = self.model_image_size != (None, None)
		self.boxes, self.scores, self.classes = self.generate()

	def _get_class(self):
		classes_path = os.path.expanduser(self.classes_path)
		with open(classes_path) as f:
			class_names = f.readlines()
		class_names = [c.strip() for c in class_names]
		return class_names

	def _get_anchors(self):
		anchors_path = os.path.expanduser(self.anchors_path)
		with open(anchors_path) as f:
			anchors = f.readline()
			anchors = [float(x) for x in anchors.split(',')]
			anchors = np.array(anchors).reshape(-1, 2)
		return anchors

	def generate(self):
		assert self.weights_path.endswith('.hdf5'), 'Making sure you got the right standard of weights file extension.'

		self.yolo_model, _ = yolo_training(num_classes=len(self.class_names), weights_path=self.weights_path, freeze_body='all')

		# Generate colors for drawing bounding boxes.
		hsv_tuples = [(x / len(self.class_names), 1., 1.)
              		for x in range(len(self.class_names))]
		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		self.colors = list(
    		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        		self.colors))
		random.seed(10101)  # Fixed seed for consistent colors across runs.
		random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
		random.seed(None)  # Reset seed to default.

		# Generate output tensor targets for filtered bounding boxes.
		self.input_image_shape = K.placeholder(shape=(2, ))
		boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
        		len(self.class_names), self.input_image_shape,
        		score_threshold=self.score, iou_threshold=self.iou)
		return boxes, scores, classes

	def detect(self, image, generate_data=False):
		start = time.time()

		if self.is_fixed_size:
			assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
			assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
			boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
		else:
			new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
			boxed_image = letterbox_image(image, new_image_size)
		image_data = np.array(boxed_image, dtype='float32')

		image_data /= 255.
		image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

		out_boxes, out_scores, out_classes = self.sess.run(
            		[self.boxes, self.scores, self.classes],
            		feed_dict={
                		self.yolo_model.input: image_data,
                		self.input_image_shape: [image.size[1], image.size[0]],
                		K.learning_phase(): 0
            		})

		font = ImageFont.truetype(font=os.path.join(os.path.dirname(__file__), '..', 'font', 'FiraMono-Medium.otf'),
            		size=np.floor(2.3e-2 * image.size[1] + 0.5).astype('int32'))
		thickness = (image.size[0] + image.size[1]) // 350
		data = ''
		for i, c in reversed(list(enumerate(out_classes))):
			predicted_class = self.class_names[c]
			box = out_boxes[i]
			score = out_scores[i]
			label = '{} {:.2f}'.format(predicted_class, score)
			draw = ImageDraw.Draw(image)
			label_size = draw.textsize(label, font)

			top, left, bottom, right = box
			top = max(0, np.floor(top + 0.5).astype('int32'))
			left = max(0, np.floor(left + 0.5).astype('int32'))
			bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
			right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
			if generate_data:
				data += ' {},{},{},{},{}'.format(left, top, right, bottom, c)
				continue
			if top - label_size[1] >= 0:
				text_origin = np.array([left, top - label_size[1]])
			else:
				text_origin = np.array([left, top + 1])

			# My kingdom for a good redistributable image drawing library.
			for j in range(thickness):
				draw.rectangle(
    					[left + j, top + j, right - j, bottom - j],
    					outline=self.colors[c]
				)
			draw.rectangle(
                		[tuple(text_origin), tuple(text_origin + label_size)],
				fill=self.colors[c])
			draw.text(text_origin, label, fill=(0, 0, 0), font=font)
			del draw

		if generate_data:
			return data

		end = time.time()
		return image



	def detect_video(self, video_path, generate_data=False):

		vid = cv2.VideoCapture(video_path)
		frame_w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

		if not vid.isOpened():
			print("Unable to open video!")
			return

		output_path = input("Input save path for detected video (strictly mp4 video format, 'skip' to skip saving): ")
		if output_path == 'skip' or not output_path.endswith('.mp4'):
			print('Skip saving this video.')
			output_path = None

		if output_path is not None:
			try:
				out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (frame_w, frame_h))
			except:
				print("Unable to save video.")
				output_path = None

		accum_time = 0
		curr_fps = 0
		fps = "FPS: ??"
		prev_time = timer()

		while vid.isOpened():
			return_value, frame = vid.read()
			if frame is None:
				cv2.destroyAllWindows()
				break
			image = Image.fromarray(frame)
			image = self.detect(image)
			result = np.asarray(image)
			curr_time = timer()
			exec_time = curr_time - prev_time
			prev_time = curr_time
			accum_time = accum_time + exec_time
			curr_fps = curr_fps + 1
			if accum_time > 1:
				accum_time = accum_time - 1
				fps = "FPS: " + str(curr_fps)
				curr_fps = 0
			cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            			fontScale=0.50, color=(255, 0, 0), thickness=2)
			cv2.namedWindow("result", cv2.WINDOW_NORMAL)
			cv2.imshow("result", result)
			if output_path is not None:
				out.write(result)
			elif cv2.waitKey(1) & 0xFF == ord('s'):
				cv2.destroyAllWindows()
				out.release()
				vid.release()
				break


	def detect_image(self, img):
		try:
			image = Image.open(img)
		except:
			print('Error encountered trying to open this image file.')
		else:
			r_image = self.detect(image)
			image = np.array(r_image, dtype='uint8')
			cv2.namedWindow(img, cv2.WINDOW_NORMAL)
			cv2.imshow(img, image)
			while True:
				if cv2.waitKey(0):
					cv2.destroyAllWindows()
					break
			saved = input('Input path to save this image("skip" to skip saving): ')
			if saved != 'skip':
				try:
					cv2.imwrite(saved, image)
				except:
					print('Invalid input path, skip saving!')

	def close_session(self):
		self.sess.close()


def main(args=None):
	image_extension = ['.png', '.jpg', '.jpeg', '.tif']
	video_extension = ['.mp4', '.avi', '.mov', '.flv', '.wmv', '.MP4']
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)
	yolo = YOLO(args)
	if args.generate_data:
		opt = input('Enter [I/V] to specify whether to generate data from images or videos: ').lower()
		assert opt in 'iv', 'Invalid input.'

		if opt == 'i':
			images_dir = input('Images directory name to generate data: ')
			images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'generated_data', images_dir)
			assert os.path.isdir(images_path), 'Invalid images path.'
			data = ''
			for i, image in enumerate(os.listdir(images_path)):
				print('\rImage detecting: {}/{}'.format(i, len(os.listdir(images_path))), end='')
				img = Image.fromarray(cv2.imread(os.path.join(images_path, image)))
				data = data + '{} {},{}{}\n'.format(image, img.size[0], img.size[1], yolo.detect(img, generate_data=True))
			with open(images_path + '.txt', 'a') as file:
				file.write(data)
			output_path = images_path
			output_dir = images_dir
		elif opt == 'v':
			videos_dir = input('Videos directory name to generate data: ')
			videos_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'videos', videos_dir)
			assert os.path.isdir(videos_path), 'Invalid videos path.'
			output_dir = input('Output directory name to save generated data: ')
			output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'generated_data', output_dir)
			if os.path.exists(output_path):
				shutil.rmtree(output_path)
			os.mkdir(output_path)
			for video in os.listdir(videos_path):
				print('Video:', video)
				data = ''
				video_file = os.path.join(videos_path, video)
				if video_file[-4:] not in video_extension:
					continue
				cap = cv2.VideoCapture(video_file)
				frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				gap = 50
				i = 0
				while cap.isOpened():
					ret, frame = cap.read()
					if i % gap == 0:
						if frame is None:
							break
						image = Image.fromarray(frame)
						data = data + '{}{}.jpg {},{}{}\n'.format(video[:-4], i, frame_w, frame_h, yolo.detect(image, generate_data=True))
						output_file = os.path.join(output_path, '{}{}.jpg'.format(video[:-4], i))
						print('Output image filename:', output_file)
						cv2.imwrite(output_file, frame)
					i += 1

				with open(output_path + '.txt', 'a') as file:
					file.write(data)
		categories = {i: line.strip('\n') for i, line in enumerate(open(yolo.classes_path).readlines())}
		yolo.close_session()
		databaseSrc = 'YOLOv3-labeled'
		textfile = output_path + '.txt'
		foldername = output_dir
		lines = None
		with open(textfile, 'r') as file:
	                lines = [line.strip('\n') for line in file.readlines()]
		for line in lines:
			line = line.split()
			filename = line[0][:-4]
			localImgPath = os.path.join(output_path, filename)
			imgSize = tuple(int(i) for i in line[1].split(',')) + (3,)
			writer = PascalVocWriter(foldername, filename, imgSize, databaseSrc, localImgPath)
			for bbox in line[2:]:
				bbox = [int(c) for c in bbox.split(',')]
				writer.addBndBox(bbox[0], bbox[1], bbox[2], bbox[3], categories[bbox[4]], 0)
			targetFile = os.path.join(output_path, filename + '.xml')
			writer.save(targetFile)
			del writer
		os.remove(textfile)
		sys.exit()

	while True:
		original = input('Input image or video path("exit" to exit program):').lower()
		if original == 'exit':
			yolo.close_session()
			break
		if original[-4:] in video_extension:
			yolo.detect_video(original)
		elif original[-4:] in image_extension:
			yolo.detect_image(original)
		else:
			print("Invalid input!!!")


if __name__ == '__main__':
	main()
