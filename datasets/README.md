By default it is expected that this directory will hold all of the datasets.

To download datasets use the `get_?_dataset.sh` scripts running them from within the `datasets` dir:
``` bash
VidDet/datasets$ . get_get_voc_dataset.sh
VidDet/datasets$ . get_get_coco_dataset.sh
VidDet/datasets$ . get_get_imgnetdet_dataset.sh
VidDet/datasets$ . get_get_imgnetvid_dataset.sh
```

These will make new directories (shown below) and download into. If you want to use
_symbolically linked_ directories you will need to make these prior to
running the scripts, **ensuring they have have the exact spelling as below**:

```
VidDet/datasets/PascalVOC/
VidDet/datasets/MSCoco/
VidDet/datasets/ImageNetDET/
VidDet/datasets/ImageNetVID/
```

If using an IDE such as PyCharm be sure to mark each of the
dataset directories as `excluded` so they aren't indexed. Indexing just
takes too long for the many files that will exist in these directories.