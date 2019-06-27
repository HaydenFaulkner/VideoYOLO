#!/bin/bash

mkdir ImageNetDET
cd ImageNetDET

# Download and extract the data
wget -c http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET.tar.gz
wget -c http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET_test_new.tar.gz

tar xzf ILSVRC2017_DET.tar.gz
tar xzf ILSVRC2017_DET_test_new.tar.gz

cd ..