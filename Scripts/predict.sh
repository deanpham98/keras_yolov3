cd ../main
python predict.py --weights-file new.hdf5 \
		  --anchors-file coco_anchors.txt \
		  --classes-file classes.txt \
		  --score 0.3 \
		  --iou 0.5
