By default it is expected that this directory will hold all of the datasets.

To download datasets you can either use the `get_?_dataset.sh` scripts or 
the python `download_datasets.py` script:
``` bash
get_voc_dataset.sh
get_coco_dataset.sh
get_imgnetdet_dataset.sh
get_imgnetvid_dataset.sh
```
or
``` bash
python download_datasets.py --dataset voc
python download_datasets.py --dataset coco
python download_datasets.py --dataset det
python download_datasets.py --dataset vid
```

Both will will make new directories
herein (shown below) and download into. If you want to use 
_symbolically linked_ directories you will need to make these prior to
running the scripts, **ensuring they have have the exact spelling as below**:

```
datasets/PascalVOC/
datasets/MSCoco/
datasets/ImageNetDET/
datasets/ImageNetVID/
```

If using an IDE such as PyCharm be sure to mark each of the
dataset directories as `excluded` so they aren't indexed. Indexing just
takes too long for the many files that will exist in these directories.