cd $(dirname "$0")/../main/
python train.py --batch-size 32 --steps 1900 --epochs 30 --training-data bdd_train.txt --validation-data bdd_val.txt --classes-file bdd_classes.txt --anchors-file bdd_anchors.txt --weights-path epoch_9_21_06_2018_03_33_02.weights --tensorboard --no-mAP
