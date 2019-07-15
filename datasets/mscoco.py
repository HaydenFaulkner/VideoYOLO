"""MS COCO object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import mxnet as mx
from gluoncv.data.mscoco.utils import try_import_pycocotools
from gluoncv.data.base import VisionDataset
from gluoncv.utils.bbox import bbox_xywh_to_xyxy, bbox_clip_xyxy

__all__ = ['COCODetection']


class COCODetection(VisionDataset):
    """MS COCO detection dataset.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/coco'
        Path to folder storing the dataset.
    splits : list of str, default ['instances_val2017']
        Json annotations name.
        Candidates can be: instances_val2017, instances_train2017.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    min_object_area : float
        Minimum accepted ground-truth area, if an object's area is smaller than this value,
        it will be ignored.
    skip_empty : bool, default is True
        Whether skip images with no valid object. This should be `True` in training, otherwise
        it will cause undefined behavior.
    use_crowd : bool, default is True
        Whether use boxes labeled as crowd instance.

    """

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('instances_val2017',), transform=None, min_object_area=0,
                 skip_empty=True, use_crowd=True, inference=False):
        super(COCODetection, self).__init__(root)
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._min_object_area = min_object_area
        self._skip_empty = skip_empty
        self._use_crowd = use_crowd
        self._inference = inference
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        # to avoid trouble, we always use contiguous IDs except dealing with cocoapi
        self.index_map = dict(zip(self.classes, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._coco = []
        self._items, self._labels = self._load_jsons()

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats()[0] + '\n'

    @property
    def coco(self):
        """Return pycocotools object for evaluation purposes."""
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        if len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files. \
                Please use single JSON dataset and evaluate one by one".format(len(self._coco)))
        return self._coco[0]

    @property
    def classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/coco.names')
        with open(names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # type(self).CLASSES = classes
        # return type(self).CLASSES
        return classes

    @property
    def classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/coco.names')
        with open(names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    @property
    def wn_classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/coco_wn.names')
        with open(names, 'r') as f:
            wn_classes = [line.strip() for line in f.readlines()]
        return wn_classes

    @property
    def annotation_dir(self):
        """
        The subdir for annotations. Default is 'annotations'(coco default)
        For example, a coco format json file will be searched as
        'root/annotation_dir/xxx.json'
        You can override if custom dataset don't follow the same pattern
        """
        return 'annotations'

    @property
    def image_ids(self):
        return self.coco.getImgIds()

    def _parse_image_path(self, entry):
        """How to parse image dir and path from entry.

        Parameters
        ----------
        entry : dict
            COCO entry, e.g. including width, height, image path, etc..

        Returns
        -------
        abs_path : str
            Absolute path for corresponding image.

        """
        dirname, filename = entry['coco_url'].split('/')[-2:]
        abs_path = os.path.join(self._root, dirname, filename)
        return abs_path

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = np.array(self._labels[idx])
        img = mx.image.imread(img_path, 1)
        if self._inference:
            if self._transform is not None:
                return self._transform(img, label, idx)
            return img, label, idx
        else:
            if self._transform is not None:
                return self._transform(img, label)
            return img, label

    def sample_path(self, idx):
        return self._items[idx]

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self._root, self.annotation_dir, split) + '.json'
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous

            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                abs_path = self._parse_image_path(entry)
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label = self._check_load_bbox(_coco, entry)
                if not label:
                    continue
                items.append(abs_path)
                labels.append(label)
        return items, labels

    def _check_load_bbox(self, coco, entry):
        """Check and load ground-truth labels"""
        entry_id = entry['id']
        # fix pycocotools _isArrayLike which don't work for str in python3
        entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
        ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj['area'] < self._min_object_area:
                continue
            if obj.get('ignore', 0) == 1:
                continue
            if not self._use_crowd and obj.get('iscrowd', 0):
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] > 0 and xmax > xmin and ymax > ymin:
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                valid_objs.append([xmin, ymin, xmax, ymax, contiguous_cid])
        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append([-1, -1, -1, -1, -1])
        return valid_objs

    def image_size(self, id):
        entry = self.coco.loadImgs(id)[0]
        return entry['width'], entry['height']

    def stats(self):
        cls_boxes = []
        n_samples = len(self._labels)
        n_boxes = [0]*len(self.classes)
        for label in self._labels:
            for box in label:
                n_boxes[box[4]] += 1

        out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n'.format('Split:', ', '.join(self._splits),
                                                                                    'Images:', n_samples,
                                                                                    'Boxes:', sum(n_boxes),
                                                                                    'Classes:', len(self.classes))
        out_str += '-'*35 + '\n'
        for i in range(len(n_boxes)):
            out_str += '{0: <3} {1: <10} {2: <15} {3}\n'.format(i, self.wn_classes[i], self.classes[i], n_boxes[i])
            cls_boxes.append([i, self.wn_classes[i], self.classes[i], n_boxes[i]])
        out_str += '-'*35 + '\n'

        return out_str, cls_boxes


if __name__ == '__main__':
    train_dataset = COCODetection(
        root=os.path.join('datasets', 'MSCoco'), splits='instances_train2017', use_crowd=False)
    val_dataset = COCODetection(
        root=os.path.join('datasets', 'MSCoco'), splits='instances_val2017', skip_empty=False)

    print(train_dataset)
    print(val_dataset)
