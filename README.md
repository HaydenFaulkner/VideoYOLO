# VidDet
A fast, accurate and diverse object detection pipeline for video written
in [MXNet](https://mxnet.apache.org/) and [GluonCV](https://gluon-cv.mxnet.io/).

## Todo
- Train and upload pre-trained models for all datasets and MobileNet
backbone
- Add temporal processing models

## Datasets

We support training and testing with the following datasets:
- [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
- [MSCoco](http://cocodataset.org/#download)
- [ImageNetDET](http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php)
- [ImageNetVID](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php)


| Dataset     |       split      | Images (Videos) |  Boxes (Obj Instances) | Categories |
|-------------|------------------|-----------------|----------------|------------|
| PascalVOC   | `trainval 07+12` |           16551 |          47223 |         20 |
| PascalVOC   |     `test 07`    |            4952 |          14976 |         20 |
|             |                  |                 |                |            |
| MSCoco      |    `train 17`    |          117266 |         849901 |         80 |
| MSCoco      |     `val 17`     |            5000 |          36828 |         80 |
|             |                  |                 |                |            |
| ImageNetDET |     `train`      |          456567 |         478806 |        200 |
| ImageNetDET |       `val`      |           20121 |          55502 |        200 |
| ImageNetDET | `train_nonempty` |          333474 |         478806 |        200 |
| ImageNetDET |  `val_nonempty`  |           18680 |          55502 |        200 |
|             |                  |                 |                |            |
| ImageNetVID |    `train15`     |  1122397 (3862) | 1731913 (7911) |         30 |
| ImageNetVID |      `val15`     |    176126 (555) |  273505 (1309) |         30 |
| ImageNetVID |     `test15`     |    315176 (937) |             NA |         30 |
| ImageNetVID |`train15_nonempty`|  1086132 (3862) | 1731913 (7911) |         30 |
| ImageNetVID | `val15_nonempty` |    172080 (555) |  273505 (1309) |         30 |
|             |                  |                 |                |            |
| ImageNetVID |    `train17`     |  1181113 (4000) | 1859625 (8394) |         30 |
| ImageNetVID |      `val17`     |   512360 (1314) |  795433 (3181) |         30 |
| ImageNetVID |     `test17`     |   765631 (2000) |             NA |         30 |
| ImageNetVID |`train17_nonempty`|  1142945 (4000) | 1859625 (8394) |         30 |
| ImageNetVID | `val17_nonempty` |   492183 (1314) |  795433 (3181) |         30 |
| ImageNetVID | `train17_ne_0.04`|    47481 (4000) |   78501 (8682) |         30 |
| ImageNetVID |  `val17_ne_0.04` |    20353 (1314) |   33384 (3295) |         30 |


See [datasets](/datasets/) for downloading and organisation information...

## Models
Currently:
- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

See [models](/models/) for downloading and organisation information...

## Installation

#### Pip

```bash
pip install -r requirements.txt
```

#### Conda

```bash
conda env create -f environment.yml
conda activate viddet-mx
```

## Usage

### Training
To train a model you can use something like:
```
python train_yolov3.py --dataset voc --gpus 0,1,2,3 --save_prefix 0001 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```

If you don't have this much power available you will need to specify a lower batch size:
```
python train_yolov3.py --batch_size 4 --dataset voc --save_prefix 0001 --warmup_lr 0.0001 --warmup_epochs 3
```

Using MobileNet1.0 `--network mobilenet1.0` and batch size of 16
`--batch_size 16` for `voc` uses **9GB** of GPU memory

***We found a warmup was necessary for YOLOv3***

### Finetuning
To finetune a model you need to specify a `--resume` path and a `--trained_on` dataset:
```
python train_yolov3.py --dataset voc --trained_on coco --resume models/0003/yolo3_darknet53_coco_best.params --gpus 0,1,2,3 --save_prefix 0006 --num_workers 16 --warmup_lr 0.0001 --warmup_epochs 3 --syncbn True
```

### Detection, Testing & Visualisation
To evaluate a model you can use something like:
```
python detect_yolov3.py --batch_size 1 --pretrained models/0001/yolo3_darknet53_voc_best.params --metrics voc --dataset voc --save_prefix 0001
```

You can also evaluate on different data than the model was trained on
(voc trained model on vid set):
```
python detect_yolov3.py --batch_size 1 --pretrained models/0001/yolo3_darknet53_voc_best.params --metrics voc,coco,vid --dataset vid --save_prefix 0001
```

Visualisation is **off** by default use `--visualise True` to write out images with boxes displayed.

## Results
| Model  | Trained On | Tested On | VOC<sub>12</sub> | AP<sub>.5-.95</sub> | AP<sub>.5 | AP<sub>.75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|--------|------------|-----------|------------------|---------------------|-----------|------------------|----------------|----------------|----------------|
| `0001` | VOC `trainval 07+12` | VOC `test 07` | .835 | .463 | .733 | .510 | .118| .317 | .559 |
