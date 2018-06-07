cd $(dirname "$0")/../main/
python train.py --no-transform --no-enhance --batch-size 16 --steps 312 --epochs 20 --training-data voc_train.txt --validation-data 2007_train.txt --classes-file voc_classes.txt --tensorboard

