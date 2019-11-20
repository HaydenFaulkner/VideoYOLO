#!/bin/bash

# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh
mkdir MSCoco
cd MSCoco

# Clone COCO API
git clone https://github.com/pdollar/coco
mv coco cocoapi

# Download and Unzip Images 2014 then 2017
mkdir images
cd images
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

unzip -q train2014.zip
unzip -q val2014.zip
unzip -q test2014.zip

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip

cd ..


# Download COCO Metadata
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip -q annotations_trainval2014.zip
unzip -q annotations_trainval2017.zip
unzip -q image_info_test2017.zip

cd ..