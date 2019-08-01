#!/bin/bash

mkdir YouTubeBB
cd YouTubeBB

# Download and extract the annotation data
# (the videos need to be taken from YouTube directly using the python functions in the youtubebb.py file)
wget -c https://research.google.com/youtube-bb/yt_bb_classification_train.csv.gz
wget -c https://research.google.com/youtube-bb/yt_bb_classification_validation.csv.gz
wget -c https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz
wget -c https://research.google.com/youtube-bb/yt_bb_detection_validation.csv.gz

gunzip -q yt_bb_classification_train.csv.gz
gunzip -q yt_bb_classification_validation.csv.gz
gunzip -q yt_bb_detection_train.csv.gz
gunzip -q yt_bb_detection_validation.csv.gz

cd ..