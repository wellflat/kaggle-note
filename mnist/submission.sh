#!/bin/sh


#kaggle competitions download -c digit-recognizer

msg="first submit, apply lenet"
submit_file=./submission/submission1.csv
kaggle competitions submit -c digit-recognizer -f $submit_file -m "$msg"
