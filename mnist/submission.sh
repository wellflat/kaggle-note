#!/bin/sh


#kaggle competitions download -c digit-recognizer

msg="apply normalize"
submit_file=./submission/submission3.csv
kaggle competitions submit -c digit-recognizer -f $submit_file -m "$msg"
