"""ImageNet VID object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
from tqdm import tqdm
import warnings
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from gluoncv.data.base import VisionDataset


class ImageNetVidDetection(VisionDataset):
    """ImageNet VID detection Dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/vidt'
        Path to folder storing the dataset.
    splits : tuple, default ('train')
        Candidates can be: 'train', 'val', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 30 classes are mapped into indices from 0 to 29. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    """

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'det'),
                 splits=('train',), frames=True,
                 transform=None, index_map=None):
        super(ImageNetVidDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._frames = frames
        self._anno_path = os.path.join('{}', 'Annotations', 'VID', '{}', '{}.xml')
        self._image_path = os.path.join('{}', 'Data', 'VID', '{}', '{}.JPEG')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._items = self._load_items(splits)

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/imagenetvid_wn.names')
        with open(names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        type(self).CLASSES = classes
        try:
            self._validate_class_names(type(self).CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        if self._frames:
            img_id = self._items[idx]
            img_path = self._image_path.format(*img_id)
            label = self._load_label(idx)
            img = mx.image.imread(img_path, 1)
            if self._transform is not None:
                return self._transform(img, label)
            return img, label
        else:
            raise NotImplementedError

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for split in splits:
            root = self._root  # os.path.join(self._root, 'ILSVRC')
            lf = os.path.join(root, 'ImageSets', 'VID', split + '.txt')

            print("Loading splits from: {}".format(lf))
            with open(lf, 'r') as f:
                ids_ = [(root, split, line.split()[0]) for line in f.readlines()]

            ids += ids_

        if not self._frames:
            raise NotImplementedError

        return ids

    def _load_label(self, idx, items=None):
        """Parse xml file and return labels."""
        if items:  # used to process items before they are actually assigned to self
            img_id = items[idx]
        else:
            img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        if not os.path.exists(anno_path):
            return np.array([])
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)  # we dont need to minus 1 here as the already 0 > h/w
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id])

        if len(label) < 1:
            label.append([-1,-1,-1,-1,-1])
        return np.array(label)

    @staticmethod
    def _validate_label(xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    @staticmethod
    def _validate_class_names(class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _verify_nonempty_annotations(self, ids):
        """Checks annotations and returns only those that contain at least one box, as well as a lil stats str"""
        good_ids = []
        removed = 0
        nboxes = 0
        for idx in tqdm(range(len(ids)), desc="Removing images that have 0 boxes"):
            nb = len(self._load_label(idx, items=ids))
            if nb < 1:
                removed += 1
            else:
                nboxes += nb
                good_ids.append(ids[idx])

        str_ = "Removed {} out of {} images, leaving {} with {} boxes over {} classes.\n".format(
            removed, len(ids), len(good_ids), nboxes, len(self.classes))

        return good_ids, str_
