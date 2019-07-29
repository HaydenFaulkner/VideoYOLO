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

## Our Models
Our models, log files, and evaluation results are available for download
by clicking on each model ID below.

#### 0001
[**Download**](http://hf.id.au/models/VidDet/0001.tar.gz)

Trained on [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) `trainval 07+12`

```
python train_yolov3.py --dataset voc --gpus 0,1,2,3 --save_prefix 0001 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```
