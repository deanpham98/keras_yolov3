cd $(dirname "$0")/../main/
python train.py --batch-size 32 \
		--tensorboard \
		--steps 2500 \
		--epochs 100 \
		--initial-epoch 0 \
		--training-data train2014.txt \
		--validation-data val2014.txt \
		--classes-file classes.txt \
		--anchors-file coco_anchors.txt \
		--weights-path new.hdf5
