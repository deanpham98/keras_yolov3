import os
import sys
import cv2
import numpy as np 
from copy import deepcopy

labels = {'car': 0, 'truck': 1, 'traffic_light': 2, 'bus': 3, 'motorcycle': 4, 'person': 5}
rect = [0, 0, 0, 0]
start_point = [0, 0]
end_point = [0, 0]
clicked = False
label = ''

def onMouse(event, x, y, flags, params):
	global rect, start_point, end_point, clicked

	if event == cv2.EVENT_LBUTTONDOWN:
		clicked = True
		start_point[0] = x
		start_point[1] = y
		end_point[0] = x
		end_point[1] = y
	elif event == cv2.EVENT_LBUTTONUP:
		end_point[0] = x
		end_point[1]= y
		clicked = False
	elif event == cv2.EVENT_MOUSEMOVE:
		if clicked:
			end_point[0] = x
			end_point[1] = y

	if clicked:
		if start_point[0] <= end_point[0] and start_point[1] <= end_point[1]:
			rect[0] = start_point[0]
			rect[1] = start_point[1]
			rect[2] = end_point[0]
			rect[3] = end_point[1]
			current_image = deepcopy(image)
			cv2.rectangle(current_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
			cv2.imshow("image", current_image)
		else:
			cv2.imshow("image", image)
	else:
		if start_point[0] <= end_point[0] and start_point[1] <= end_point[1]:
			rect[0] = start_point[0]
			rect[1] = start_point[1]
			rect[2] = end_point[0]
			rect[3] = end_point[1]		


if __name__ == "__main__":
	for file in os.listdir("images"):
		print("Label image file: " + file)
		data = ['./get_data/images/' + file]
		image = cv2.imread(os.path.join("images", file))
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", onMouse)
		while True:
			key = cv2.waitKey(0) & 0xFF
			if key == ord('a'):
				data.append('{},{},{},{},{}'.format(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), label))
				print("Detected object at: " + data[-1])
				cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
				cv2.imshow("image", image)
				rect = [0, 0, 0, 0]
				start_point = [0, 0]
				end_point = [0, 0]
				clicked = False
				label = ''
			if key == ord('l'):
				label = input('Enter the class of the object: ')
				print("Temporarily label object as: " + label)
				label = labels[label]
			if key == ord('s'):
				with open("annotation.txt", "a+") as saved_file:
					saved_file.write(" ".join(data) + '\n')
				print("Saving all labels of the image {} to annotation.txt, which has {}".format(file, len(data) - 1))
				cv2.destroyWindow("image")
				break
			if key == ord('r'):
				print("Resetting current detected data...")
				rect = [0, 0, 0, 0]
				start_point = [0, 0]
				end_point = [0, 0]
				clicked = False
				label = ''
			if key == ord('q'):
				print("Quit labelling program!")
				cv2.destroyWindow("image")
				sys.exit()
			if key == ord('n'):
				print("Skip this image")
				break
