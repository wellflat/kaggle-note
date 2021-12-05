#!/bin/sh


#kaggle competitions download -c digit-recognizer

msg="transplanted to lightning"
submit_file=./submission/submission6.csv
kaggle competitions submit -c digit-recognizer -f $submit_file -m "$msg"
