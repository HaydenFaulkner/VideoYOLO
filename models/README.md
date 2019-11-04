<h1 align='center'>Models</h1>

## General Structure

This directory holds the models, both those downloaded automatically
from GluonCV the ones that are trained. Trained models will be put in a
subdirectory named with the `args.save-prefix` argument. For example
our structure looks as follows:


```
VidDet/models/README.md
VidDet/models/definitions/darknet/weights/darknet53-2189ea49.params               <- downloaded by gluoncv
VidDet/models/definitions/darknet/weights/mobilenet/mobilenet1.0-efbb2ca3.params  <- downloaded by gluoncv

VidDet/models/experiments/0001/                                                   <- our own trained models

```

## GluonCV ModelZoo
GluonCV provides a few pre-trained models in their
[Model Zoo](https://gluon-cv.mxnet.io/model_zoo/detection.html). Such
models are downloaded automatically when specified in GluonCV with the
appropriate `gluoncv.model_zoo.get_model()` function call, *however*
we present these models for download and organise them similarly to
our trained models.

<h5 align=center>GCV1 (0001 Alternative)   -   Trained on <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit">PascalVOC</a>   -   <a href="http://hf.id.au/models/VidDet/GCV1.tar.gz">Download</a> </h5> 

<h5 align=center>GCV2 (0003 Alternative)   -   Trained on <a href="http://cocodataset.org/#download">MSCOCO</a>   -   <a href="http://hf.id.au/models/VidDet/GCV2.tar.gz">Download</a> </h5> 


## Our Models
Our models, log files, and evaluation results are available for download
by clicking on each model ID below.

<h5 align=center>0001   -   Trained on <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit">PascalVOC</a>   -   <a href="http://hf.id.au/models/VidDet/0001.tar.gz">Download</a> </h5> 

```
python train_yolov3.py --dataset voc --gpus 0,1,2,3 --save_prefix 0001 --num_workers 16 --warmup_epochs 4 --syncbn
```

<h5 align=center>0002   -   Trained on <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNetDET</a>   -   <a href="http://hf.id.au/models/VidDet/0002.tar.gz">Download (SOON)</a> </h5> 

```
python train_yolov3.py --dataset det --gpus 0,1,2,3 --save_prefix 0002 --num_workers 16 --warmup_epochs 3 --epochs 140 --lr_decay_epoch 100,120 --syncbn
```

<h5 align=center>0003   -   Trained on <a href="http://cocodataset.org/#download">MSCOCO</a>   -   <a href="http://hf.id.au/models/VidDet/0003.tar.gz">Download</a> </h5> 

```
python train_yolov3.py --dataset coco --gpus 0,1,2,3 --save_prefix 0003 --num_workers 16 --warmup_epochs 3 --epochs 280 --lr_decay_epoch2 20,250 --syncbn
```

<h5 align=center>0004 - Trained on <a href="http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php">ImageNetVID</a>   -   <a href="http://hf.id.au/models/VidDet/0004.tar.gz">Download</a> </h5> 

```
python train_yolov3.py --dataset vid --gpus 0,1,2,3 --save_prefix 0004 --num_workers 16 --warmup_epochs 3 --epochs 280 --lr_decay_epoch 220,250 --every 25 --syncbn 
```

## Results
Evaluated with `voc` and `coco` metrics. Box Area's - **S**mall `<32`,
 **M**edium `32-96`, **L**arge `>96`

| Model  | Trained On | Tested On | VOC<sub>12</sub> | AP<sub>.5-.95</sub> | AP<sub>.5 | AP<sub>.75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|--------|------------|-----------|------------------|---------------------|-----------|------------------|----------------|----------------|----------------|
|**0001**|     VOC    |    VOC    | .835 | .463 | .733 | .510 | .118 | .317 | .559 |
|**GCV1**|     VOC    |    VOC    | .836 | .462 | .735 | .500 | .113 | .304 | .564 |
|**0003**|    COCO    |   COCO    | .525 | .288 | .515 | .296 | .136 | .306 | .427 |
|**GCV2**|    COCO    |   COCO    | .579 | .360 | .571 | .387 | .173 | .387 | .522 |
|**0004**|     VID    |    VID    | .478 | .274 | .453 | .298 | .031 | .130 | .330 |

Evaluated with `vid` metric. Box Area's - **S**mall `<50`,
 **M**edium `50-150`, **L**arge `>150`. Instance's Speed (motion IoU) -
 **SL**ow `>0.9`,  **MO**derate `0.7-0.9`, **FA**st `<0.7`

| Model  | Trained On | Tested On | mAP | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | AP<sub>SL</sub> | AP<sub>MO</sub> | AP<sub>FA</sub> |
|--------|------------|-----------|------------------|---------------------|-----------|------------------|----------------|----------------|----------------|
|**0004**|     VID    |    VID    | .454 | .136 | .328 | .555 | .522 | .442 | .292 |

