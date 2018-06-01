from generator import Generator
from utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path


def _read_classes(file):    
    return {key.strip("\n"): value for value, key in enumerate(file.readlines())}


def _read_annotations(file, classes):
    result = {}
    lines = [line.split(" ") for line in file.readlines()]
    for line in lines:
        img_file = line[0]
        line[1:] = [box.split(",") for box in line[1:]]
        result[img_file] = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3], 'class': box[4]} for box in line[1:]]
    return result

class DataGenerator(Generator):
    def __init__(
        self,
        data_file,
        class_file,
        base_dir=None,
        **kwargs
    ):
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(data_file)

        # parse the provided class file
        with open(class_file, "r") as file:
            self.classes = _read_classes(file)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        with open(data_file, "r") as file:
            self.image_data = _read_annotations(file, self.classes)

        self.image_names = list(self.image_data.keys())

        super(DataGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes.values())

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        path   = self.image_names[image_index]
        annots = self.image_data[path]
        boxes  = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = int(annot['class'])

        return boxes