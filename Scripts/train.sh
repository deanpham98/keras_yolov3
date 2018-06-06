cd ../main/
python train.py --batch-size 16 --steps 312 --epochs 20 --training-data voc_train.txt --validation-data 2012_val.txt --classes-file voc_classes.txt --tensorboard

