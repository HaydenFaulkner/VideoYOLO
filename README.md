# VidDet
A fast, accurate and diverse object detection pipeline for video written
in [MXNet](https://mxnet.apache.org/) and [GluonCV](https://gluon-cv.mxnet.io/).

**BE WARNED : STILL A WORK IN PROGRESS**

### Datasets

We support training and testing with the following datasets:
- [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
- [MSCoco](http://cocodataset.org/#download)
- [ImageNetDET](http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php)
- [ImageNetVID](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php)


| Dataset     |       split      | Images (Videos) |  Boxes  | Categories |
|-------------|------------------|-----------------|---------|------------|
| PascalVOC   | `trainval 07+12` |           16551 |   47223 |         20 |
| PascalVOC   |     `test 07`    |            4952 |   14976 |         20 |
|             |                  |                 |         |            |
| MSCoco      |    `train 17`    |          117266 |  849901 |         80 |
| MSCoco      |     `val 17`     |            5000 |   36828 |         80 |
|             |                  |                 |         |            |
| ImageNetDET |     `train`      |          456567 |  478806 |        200 |
| ImageNetDET |       `val`      |           20121 |   55502 |        200 |
| ImageNetDET | `train_nonempty` |          333474 |  478806 |        200 |
| ImageNetDET |  `val_nonempty`  |           18680 |   55502 |        200 |
|             |                  |                 |         |            |
| ImageNetVID |    `train15`     |   1122397(3862) | 1731913 |         30 |
| ImageNetVID |      `val15`     |    176126 (555) |  273505 |         30 |
| ImageNetVID |`train15_nonempty`|         1086132 | 1731913 |         30 |
| ImageNetVID | `val15_nonempty` |          172080 |  273505 |         30 |
|             |                  |                 |         |            |
| ImageNetVID |    `train17`     |  1181113 (4000) | 1859625 |         30 |
| ImageNetVID |      `val17`     |   512360 (1314) |  795433 |         30 |
| ImageNetVID |`train17_nonempty`|         1142945 | 1859625 |         30 |
| ImageNetVID | `val17_nonempty` |          492183 |  795433 |         30 |


See [datasets](/datasets/) for downloading and organisation information...

### Models
Currently:
- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)


## Usage

### Installation

#### Pip

```bash
pip install -r requirements.txt
```

#### Conda

```bash
conda env create -f environment.yml
conda activate viddet-mx
```

### Training

### Testing

### Detecting