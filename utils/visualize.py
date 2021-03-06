import os
import sys
import pdb
import random
import colorsys
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def draw_bbox(image, annotations, classes, colors):
	try:
		img = Image.fromarray(image)
	except:
		img = Image.fromarray((image*255.).astype('uint8'))

	font = ImageFont.truetype(font=os.path.join(os.path.dirname(__file__), \
					'..', 'font', 'FiraMono-Medium.otf'), \
				  size=np.floor(2.3e-2 * img.size[1] + \
					0.5).astype('int32'))

	thickness = (img.size[0] + img.size[1]) // 350

	boxes = annotations[:, 0:4].astype('int32')

	if annotations.shape[1] == 6:
		scores = annotations[:, 4].astype('float32')
		categories = annotations[:, 5].astype('int32')
	elif annotations.shape[1] == 5:
		categories = annotations[:, 4].astype('int32')

	print({classes[i]: colors[i] for i in categories})

	for i in range(annotations.shape[0]):
		x1, y1, x2, y2 = boxes[i, :]

		predicted_class = classes[categories[i]]

		if annotations.shape[1] == 6:
			score = scores[i]
			label = '{} {:.2f}'.format(predicted_class, score)
		elif annotations.shape[1] == 5:
			label = predicted_class
		draw = ImageDraw.Draw(img)

		label_size = draw.textsize(label, font)

		print(label, (x1, y1), (x2, y2))

		if y1 - label_size[1] >= 0:
			text_origin = np.array([x1, y1 - label_size[1]])
		else:
			text_origin = np.array([x1, y1 + 1])

		for j in range(thickness):
			draw.rectangle(
				[x1 + j, y1 + j, x2 - j, y2 - j],
				outline=colors[categories[i]]
			)

		draw.rectangle(
			[tuple(text_origin), tuple(text_origin + label_size)],
			fill=colors[categories[i]]
		)

		draw.text(text_origin, label, fill=(0, 0, 0), font=font)

		del draw

	img = np.asarray(img)

	return img


def draw(data, classes):

	hsv_tuples = [(x / len(classes), 1., 1.) for x in range(len(classes))]

	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

	random.seed(10101)  # Fixed seed for consistent colors across runs.
	random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	random.seed(None)  # Reset seed to default.

	for key, values in data.items():
		result = draw_bbox(values[0], values[1], classes, colors)
		cv2.namedWindow(key, cv2.WINDOW_NORMAL)
		cv2.imshow(key, result)

	while True:
		key = cv2.waitKey(0) & 0xFF
		if key == ord('n'):
			cv2.destroyAllWindows()
			break
		elif key == ord('q'):
			cv2.destroyAllWindows()
			sys.exit()


if __name__ == '__main__':
	print('Visualization module.')
