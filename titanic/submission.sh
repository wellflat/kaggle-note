#!/bin/sh


#kaggle competitions download -c titanic

msg="feature select"
submit_file=./submission/submission.csv
kaggle competitions submit -c titanic -f $submit_file -m $msg