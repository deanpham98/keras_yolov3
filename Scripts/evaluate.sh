cd $(dirname "$0")/../main/
python evaluate.py --evaluation-data bdd_val.txt --classes-file bdd_classes.txt --weights-path epoch_4_21_06_2018_13_02_08.weights

