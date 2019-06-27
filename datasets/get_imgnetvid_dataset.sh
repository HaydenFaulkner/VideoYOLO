#!/bin/bash

mkdir ImageNetVID
cd ImageNetVID

# Download and extract the data
wget -c http://bvisionweb1.cs.unc.edu/ILSVRC2017/ILSVRC2017_VID.tar.gz
wget -c http://bvisionweb1.cs.unc.edu/ILSVRC2017/ILSVRC2017_VID_test.tar.gz
wget -c http://bvisionweb1.cs.unc.edu/ILSVRC2017/ILSVRC2017_VID_new.tar.gz

tar xzf ILSVRC2017_VID.tar.gz
tar xzf ILSVRC2017_VID_test.tar.gz
tar xzf ILSVRC2017_VID_new.tar.gz

cd ..