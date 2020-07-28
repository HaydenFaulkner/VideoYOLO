<h1 align="center">Datasets</h1>
<p align="center">
The currently supported datasets are - <a href="http://host.robots.ox.ac.uk/pascal/VOC/">Pascal VOC</a>, <a href="http://cocodataset.org/">MS-COCO</a>, <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-DET</a> and <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-VID</a>
</p>

<p align="center">The datasets should be stored in the following directory structure</p>
<pre>
VidDet/
└── datasets/
    ├── ImageNetDET (170.8 GB)
    ├── ImageNetVID (409.9 GB)
    ├── MSCoco (84.9 GB)
    ├── PascalVOC (9.8 GB)
    └── # version controlled files
</pre>

<p align="center">The datasets can be downloaded from my <a href="https://drive.google.com/drive/folders/1x79iF5-pRow7i5-R4qX09XEdN-VOgV5e?usp=sharing">Google Drive</a>:
<ul>
    <li><a href="https://drive.google.com/drive/folders/1Y3K6tWtRSM3LiadXRTsZuBOPULccIovf?usp=sharing">PascalVOC (07 + 12)</a></li>
    <li><a href="https://drive.google.com/drive/folders/1xIsUUwSIABrI5yhrTVB4P248ysghtq4t?usp=sharing">MSCoco</a></li>
    <li><a href="https://drive.google.com/drive/folders/11Ryza3GNCUK-HxKCEJv6P0yRF6tuW94o?usp=sharing">ImageNet-DET</a></li>
    <li><a href="https://drive.google.com/drive/folders/1uyIgrlQAdeCUcKiMHHzRpt795dpMvFIH?usp=sharing">ImageNet-VID</a></li>
</ul>

<h2 align="center"></h2>
<h2 align="center">A Combined Dataset</h2>
<p align="center">It's possible to combine all four datasets into one larger dataset with the utilisation of the <a href="https://github.com/HaydenFaulkner/VidDet/blob/ba28d3bf082c9e74a769bd2f1d7df47626e46b23/datasets/combined.py#L16"><code>CombinedDetection()</code></a> dataset specified in <a href="combined.py"><code>combined.py</code></a></p>

<p align="center">Following ideas from <a href="https://github.com/philipperemy/yolo-9000">YOLO-9k</a> with utilising the <a href="https://wordnet.princeton.edu/">WordNet</a> structure classes have been manually matched across datasets, furthermore a <b>hierarchical tree structure</b> has been generated for the classes. This is visualised below and is specified in <a href="trees"><code>trees/</code></a>, with the main tree (inclusive of <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-DET</a>) specified in <a href="trees/filtered_det.tree"><code>trees/filtered_det.tree</code></a></p>

<p align="center"><img src="../img/filtered_tree_det.svg"></p>

<h2 align="center"></h2>
<h2 align="center">Stats</h2>
<p align="center">These are the <code>training</code> split statistics, also samples in <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-VID</a> are calculated on a clip basis not a frame basis</p>
<p align="center"><img src="../img/viddet_class_counts_train_vids.svg"></p>
