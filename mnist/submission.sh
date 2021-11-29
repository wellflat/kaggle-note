#!/bin/sh


#kaggle competitions download -c digit-recognizer

msg="data augmentation, using Adam"
submit_file=./submission/submission5.csv
kaggle competitions submit -c digit-recognizer -f $submit_file -m "$msg"
