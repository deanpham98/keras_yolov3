import os
import xml.etree.ElementTree as ET


sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["bicycle", "bus", "car", "motorbike", "person"]


def convert_annotation(wd, year, image_id, list_file):
    in_file = open(os.path.join(wd, 'VOC/VOC%s/Annotations/%s.xml'%(year, image_id)))
    tree=ET.parse(in_file)
    root = tree.getroot()
    image = 'VOC/VOC%s/JPEGImages/%s.jpg'%(year, image_id)
    data = ''
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        data = data + " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
    if data != '':
        list_file.write(image + data + '\n')

wd = os.path.join(os.path.dirname(__file__), '..', 'data')
out = os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations')

for year, image_set in sets:
    image_ids = open(os.path.join(wd, 'VOC/VOC%s/ImageSets/Main/%s.txt'%(year, image_set))).read().strip().split()
    list_file = open(os.path.join(out, '%s_%s.txt'%(year, image_set)), 'w')
    for image_id in image_ids:
        convert_annotation(wd, year, image_id, list_file)
    list_file.close()

