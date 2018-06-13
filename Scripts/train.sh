cd $(dirname "$0")/../main/
python train.py --no-transform --batch-size 32 --steps 1500 --epochs 30 --training-data coco_train2014.txt --validation-data coco_val2014.txt --classes-file classes.txt --weights-path original.weights --tensorboard
