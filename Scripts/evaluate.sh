cd $(dirname "$0")/../main/
python evaluate.py --evaluation-data val2014.txt \
		   --classes-file classes.txt \
		   --anchors-file coco_anchors.txt \
		   --weights-path new.hdf5
