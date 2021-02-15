<h1 align='center'>VideoYOLO</h1>
<p align=center>
A fast, accurate and diverse object detection pipeline for video written
in MXNet and Gluon based on the <a href="https://pjreddie.com/darknet/yolo/">YOLOv3 network</a>
</p>

<p align="center"><img src="img/Temporal_YOLO_Conv.svg"></p>

<h2 align='center'></h2>
<h2 align='center'>Datasets</h2>

<p align="center">
The currently supported datasets are - <a href="http://host.robots.ox.ac.uk/pascal/VOC/">Pascal VOC</a>, <a href="http://cocodataset.org/">MS-COCO</a>, <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-DET</a> and <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-VID</a>. You can utilise the datasets separately or <a href="datasets#a-combined-dataset"><b>combine</b></a> them together to make a single dataset. The table below presents some of the statistics for the datasets:
</p>


<p align="center"><img src="img/viddet_data_main.svg"></p>

<!-- | Dataset     |       split      |  Images (Clips) |  Boxes (Obj Instances) | Categories | -->
<!-- |-------------|------------------|-----------------|----------------|------------| -->
<!-- | PascalVOC   | `trainval 07+12` |           16551 |          47223 |         20 | -->
<!-- | PascalVOC   |     `test 07`    |            4952 |          14976 |         20 | -->
<!-- |             |                  |                 |                |            | -->
<!-- | MSCoco      |    `train 17`    |          117266 |         849901 |         80 | -->
<!-- | MSCoco      |     `val 17`     |            5000 |          36828 |         80 | -->
<!-- |             |                  |                 |                |            | -->
<!-- | ImageNetDET |     `train`      |          456567 |         478806 |        200 | -->
<!-- | ImageNetDET |       `val`      |           20121 |          55502 |        200 | -->
<!-- | ImageNetDET | `train_nonempty` |          333474 |         478806 |        200 | -->
<!-- | ImageNetDET |  `val_nonempty`  |           18680 |          55502 |        200 | -->
<!-- |             |                  |                 |                |            | -->
<!-- | ImageNetVID |    `train15`     |  1122397 (3862) | 1731913 (7911) |         30 | -->
<!-- | ImageNetVID |      `val15`     |    176126 (555) |  273505 (1309) |         30 | -->
<!-- | ImageNetVID |     `test15`     |    315176 (937) |             NA |         30 | -->
<!-- | ImageNetVID |`train15_nonempty`|  1086132 (3862) | 1731913 (7911) |         30 | -->
<!-- | ImageNetVID | `val15_nonempty` |    172080 (555) |  273505 (1309) |         30 | -->
<!-- |             |                  |                 |                |            | -->
<!-- | ImageNetVID |    `train17`     |  1181113 (4000) | 1859625 (8394) |         30 | -->
<!-- | ImageNetVID |      `val17`     |   512360 (1314) |  795433 (3181) |         30 | -->
<!-- | ImageNetVID |     `test17`     |   765631 (2000) |             NA |         30 | -->
<!-- | ImageNetVID |`train17_nonempty`|  1142945 (4000) | 1859625 (8394) |         30 | -->
<!-- | ImageNetVID | `val17_nonempty` |   492183 (1314) |  795433 (3181) |         30 | -->
<!-- | ImageNetVID | `train17_ne_0.04`|    47481 (4000) |   78501 (8682) |         30 | -->
<!-- | ImageNetVID |  `val17_ne_0.04` |    20353 (1314) |   33384 (3295) |         30 | -->
<!-- |             |                  |                 |                |            | -->
<!-- | YouTubeBB   |      `train`     | 5608012 (301987) | 5608012 (444053) |         23 | -->
<!-- | YouTubeBB   |       `val`      |   625338 (33578) |   625338 (49193) |         23 | -->
<!-- | YouTubeBB   | `train_nonempty` | 4580762 (294853) | 4484014 (294715) |         23 | -->
<!-- | YouTubeBB   |  `val_nonempty`  |   508988 (32661) |  497616 (32650) |         23 | -->
<!-- *YouTubeBB stats are annotation stats, access to image data yet to be -->
<!-- confirmed, will be updated in future* -->

<p align="center">For more information see <a href="datasets"><code>datasets</code></a></p>

<h2 align='center'></h2>
<h2 align='center'>Models</h2>

<p align="center">For more information see <a href="models"><code>models</code></a></p>


<h2 align='center'></h2>
<h2 align='center'>Installation</h2>


<p align="center">Install <a href="https://youtube-dl.org/">youtube-dl</a> using the following command (at time of writing <code>pip install youtube-dl</code> contains bug that prevents download of videos</p>

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

<p align="center">Install via <code>pip</code> or <code>conda</code></p>
<p align="center">
<table style="width:100%">
  <tr>
    <th><code>pip</code></th>
    <th><code>conda</code></th>
  </tr>
  <tr>
    <td><pre>pip install -r requirements.txt             </pre></td>
    <td><pre>conda env create -f environment.yml           <br>conda activate viddet-mx</pre></td>
  </tr>
</table>
</p>

<h2 align='center'></h2>
<h2 align='center'>Usage</h2>

<h3 align='center'>Training</h3>

<p align="center">To train a model you can use something like:</p>

```
python train_yolov3.py --dataset voc --gpus 0,1,2,3 --save_prefix 0001 --warmup_epochs 3 --syncbn
```

<p align="center">If you don't have this much power available you will need to specify a lower batch size (this also will default to one GPU):</p>

```
python train_yolov3.py --batch_size 4 --dataset voc --save_prefix 0001 --warmup_epochs 3
```

<p align="center">.......</p>
<h3 align='center'>Finetuning</h3>

<p align="center">To finetune a model you need to specify a <code>--resume</code> path to a pretrained params model file and specify the <code>--trained_on</code> dataset, the model will be finetuned on the dataset specified with <code>--dataset</code></p>

```
python train_yolov3.py --dataset voc --trained_on coco --resume models/experiments/0003/yolo3_darknet53_coco_best.params --gpus 0,1,2,3 --save_prefix 0006 --warmup_epochs 3 --syncbn
```

<p align="center">.......</p>
<h3 align='center'>Detection, Testing & Visualisation</h3>

<p align="center">To evaluate a model you can use something like:</p>

```
python detect_yolov3.py --batch_size 1 --model_path models/experiments/0001/yolo3_darknet53_voc_best.params --metrics voc --dataset voc --save_prefix 0001
```

<p align="center">You can also evaluate on different data than the model was trained on (voc trained model on vid set):</p>

```
python detect_yolov3.py --trained_on voc --batch_size 1 --model_path models/experiments/0001/yolo3_darknet53_voc_best.params --metrics voc,coco,vid --dataset vid --save_prefix 0001
```

<p align="center">Visualisation is <b>off</b> by default add <code>--visualise</code> to write out images with boxes displayed</p>
