#!/bin/sh
cd ../utils
python extract_weights.py --filename classes.txt \
			  --new-model new.h5 \
			  --new-weights new.hdf5
