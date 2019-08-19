# Models
## General Structure
This directory holds the models, both those downloaded automatically
from GluonCV the ones that are trained. Trained models will be put in a
subdirectory named with the `args.save-prefix` argument. For example
our structure looks as follows:
```
VidDet/models/README.md
VidDet/models/darknet53-2189ea49.params        <- downloaded by gluoncv
VidDet/models/mobilenet1.0-efbb2ca3.params     <- downloaded by gluoncv

VidDet/models/0001/                            <- our own trained model

```

## GluonCV ModelZoo
GluonCV provides a few pre-trained models in their
[Model Zoo](https://gluon-cv.mxnet.io/model_zoo/detection.html). Such
models are downloaded automatically when specified in GluonCV with the
appropriate `gluoncv.model_zoo.get_model()` function call, *however*
we present these models for download and organise them similarly to
our trained models.

#### GCV1 (0001 Alternative)
[**Download**](http://hf.id.au/models/VidDet/GCV1.tar.gz)

Trained on [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) `trainval 07+12`

`yolo3_darknet53_voc` 81.5 mAP

#### GCV2 (0003 Alternative)
[**Download**](http://hf.id.au/models/VidDet/GCV2.tar.gz)

Trained on [MSCoco](http://cocodataset.org/#download) `train 17`

`yolo3_darknet53_coco` 36.0/57.2/38.7 Box AP (AP 0.5:0.95)/(AP 0.5)/(AP 0.75)

#### GCV3 (0006 Alternative)
[**Download**](http://hf.id.au/models/VidDet/GCV3.tar.gz)

Uses MobileNet1.0 instead of DarkNet53

Trained on [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) `trainval 07+12`

`yolo3_mobilenet1.0_voc` 75.8 mAP

#### GCV4 (0009 Alternative)
[**Download**](http://hf.id.au/models/VidDet/GCV4.tar.gz)

Uses MobileNet1.0 instead of DarkNet53

Trained on [ImageNetVID](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php) `train17_ne_0.04`

`yolo3_mobilenet1.0_coco` 28.6/48.9/27.8 Box AP (AP 0.5:0.95)/(AP 0.5)/(AP 0.75)


## Our Models
Our models, log files, and evaluation results are available for download
by clicking on each model ID below.

#### 0001
[**Download**](http://hf.id.au/models/VidDet/0001.tar.gz)

Trained on [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) `trainval 07+12`

```
python train_yolov3.py --dataset voc --gpus 0,1,2,3 --save_prefix 0001 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```
#### 0002
[**Download (SOON)**]()

Trained on [ImageNetDET](http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php) `train_nonempty`

```
python train_yolov3.py --dataset det --gpus 0,1,2,3 --save_prefix 0002 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```
#### 0003
[**Download**](http://hf.id.au/models/VidDet/0003.tar.gz)

Trained on [MSCoco](http://cocodataset.org/#download) `train 17`

```
python train_yolov3.py --dataset coco --gpus 0,1,2,3 --save_prefix 0003 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```

#### 0004
[**Download**](http://hf.id.au/models/VidDet/0004.tar.gz)

Trained on [ImageNetVID](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php) `train17_ne_0.04`

```
python train_yolov3.py --dataset vid --gpus 0,1,2,3 --save_prefix 0004 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True --frames 0.04
```

#### 0006
[**Download (SOON)**]()

Uses MobileNet1.0 instead of DarkNet53

Trained on [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) `trainval 07+12`

```
python train_yolov3.py --network mobilenet1_0 --dataset voc --gpus 0,1,2,3 --save_prefix 0001 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```
#### 0007
[**Download (SOON)**]()

Uses MobileNet1.0 instead of DarkNet53

Trained on [ImageNetDET](http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php) `train_nonempty`

```
python train_yolov3.py --network mobilenet1_0 --dataset det --gpus 0,1,2,3 --save_prefix 0002 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```
#### 0008
[**Download (SOON)**]()

Uses MobileNet1.0 instead of DarkNet53

Trained on [MSCoco](http://cocodataset.org/#download) `train 17`

```
python train_yolov3.py --network mobilenet1_0 --dataset coco --gpus 0,1,2,3 --save_prefix 0003 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```

#### 0009
[**Download (SOON)**]()

Uses MobileNet1.0 instead of DarkNet53

Trained on [ImageNetVID](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php) `train17_ne_0.04`

```
python train_yolov3.py --network mobilenet1_0 --dataset vid --gpus 0,1,2,3 --save_prefix 0004 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True --frames 0.04
```
