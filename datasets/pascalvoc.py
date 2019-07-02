"""Pascal VOC object detection dataset.
Copied from gluoncv
"""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import warnings
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from gluoncv.data.base import VisionDataset


class VOCDetection(VisionDataset):
    """Pascal VOC detection Dataset.
    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits=((2007, 'trainval'), (2012, 'trainval')),
                 transform=None, index_map=None, preload_label=True):
        super(VOCDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats() + '\n'

    @property
    def classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/pascalvoc.names')
        with open(names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    @property
    def wn_classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/pascalvoc_wn.names')
        with open(names, 'r') as f:
            wn_classes = [line.strip() for line in f.readlines()]
        return wn_classes

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading labels into memory...")
        return [self._load_label(idx) for idx in range(len(self))]

    def stats(self):
        n_samples = len(self._label_cache)
        n_boxes = [0] * len(self.classes)
        for label in self._label_cache:
            for box in label:
                n_boxes[int(box[4])] += 1

        out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n'.format('Split:', ', '.join([str(s[0]) + s[1] for s in self._splits]),
                                                                                    'Images:', n_samples,
                                                                                    'Boxes:', sum(n_boxes),
                                                                                    'Classes:', len(self.classes))
        out_str += '-'*35 + '\n'
        for i in range(len(n_boxes)):
            out_str += '{0: <3} {1: <10} {2: <15} {3}\n'.format(i, self.wn_classes[i], self.classes[i], n_boxes[i])
        out_str += '-'*35 + '\n'

        return out_str


if __name__ == '__main__':
    train_dataset = VOCDetection(
        root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'), splits=[(2007, 'trainval'), (2012, 'trainval')])
    val_dataset = VOCDetection(
        root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'), splits=[(2007, 'test')])

    print(train_dataset)
    print(val_dataset)
