"""MS COCO object detection dataset - Edited from GluonCV coco dataset code"""

from gluoncv.data.mscoco.utils import try_import_pycocotools
from gluoncv.data.base import VisionDataset
from gluoncv.utils.bbox import bbox_xywh_to_xyxy, bbox_clip_xyxy
import mxnet as mx
import numpy as np
import os
from tqdm import tqdm

__all__ = ['COCODetection']


class COCODetection(VisionDataset):
    """MS COCO detection dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSCoco'),
                 splits=['instances_train2017'], transform=None, min_object_area=0,
                 allow_empty=False, use_crowd=True, inference=False):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSCoco')
            splits (list): a list of splits as strings (default is ['instances_train2017'])
            transform: the transform to apply to the image/video and label (default is None)
            min_object_area (int): minimum accepted ground-truth area of box, if smaller ignored (default is 0)
            allow_empty (bool): include samples that don't have any labelled boxes? (default is False)
            use_crowd (bool): use boxes labeled as crowd instance? (default is True)
            inference (bool): are we doing inference? (default is False)
        """
        super(COCODetection, self).__init__(root)
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._min_object_area = min_object_area
        self._allow_empty = allow_empty
        self._use_crowd = use_crowd
        self._inference = inference
        self._splits = splits
        
        # to avoid trouble, we always use contiguous IDs except dealing with cocoapi
        self.index_map = dict(zip(self.classes, range(self.num_class)))
        
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._coco = []

        # load the samples and labels at once
        self.samples, self._labels = self._load_jsons()

        self.sample_ids = self.coco.getImgIds()

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
        """
        Gets a list of class names as specified in the coco.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('./datasets/names/coco.names')
        with open(names_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        return classes

    @property
    def wn_classes(self):
        """
        Gets a list of class names as specified in the coco_wn.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('./datasets/names/coco_wn.names')
        with open(names_file, 'r') as f:
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

    def _parse_image_path(self, entry):
        """
        How to parse image dir and path from entry.

        Args:
            entry (dict): COCO entry, e.g. including width, height, image path, etc..

        Returns:
            str : absolute path for corresponding image.

        """
        dirname, filename = entry['coco_url'].split('/')[-2:]
        abs_path = os.path.join(self.root, dirname, filename)
        return abs_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx (int): index of the sample in the dataset

        Returns:
            mxnet.NDArray: input image/volume
            numpy.ndarray: label
            int: idx (if inference=True)
        """
        img_path = self.sample_path(idx)
        label = np.array(self._labels[idx])
        img = mx.image.imread(img_path, 1)

        if self._transform is not None:
            return self._transform(img, label)

        if self._inference:
            return img, label, idx
        else:
            return img, label

    def sample_path(self, idx):
        return self.samples[idx]

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        samples = list()
        labels = list()
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self.root, self.annotation_dir, split) + '.json'
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
                samples.append(abs_path)
                labels.append(label)
        return samples, labels

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
            if self._allow_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append([-1, -1, -1, -1, -1])
        return valid_objs

    def image_size(self, id):
        entry = self.coco.loadImgs(id)[0]
        return entry['width'], entry['height']

    def stats(self):
        """
        Get the dataset statistics

        Returns:
            str: an output string with all of the information
            list: a list of counts of the number of boxes per class

        """
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
    train_dataset = COCODetection(splits=['instances_train2017'], use_crowd=False)
    for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
        pass
    print(train_dataset)
    val_dataset = COCODetection(splits=['instances_val2017'], allow_empty=True)
    for s in tqdm(val_dataset, desc='Test Pass of Validation Set'):
        pass
    print(val_dataset)

