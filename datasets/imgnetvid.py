"""ImageNet VID object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import json
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

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'vid'),
                 splits=('train',), allow_empty=False, frames=True,
                 transform=None, index_map=None, percent=1):
        super(ImageNetVidDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._frames = frames
        self._percent = percent
        self._allow_empty = allow_empty
        self._coco_path = os.path.join(self._root, 'jsons', '_'.join([s for s in self._splits])+'.json')
        self._anno_path = os.path.join('{}', 'Annotations', 'VID', '{}', '{}.xml')
        self._image_path = os.path.join('{}', 'Data', 'VID', '{}', '{}.JPEG')
        self.index_map = index_map or dict(zip(self.wn_classes, range(self.num_class)))
        self._items = self._load_items(splits)

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats()[0] + '\n'

    @property
    def classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/imagenetvid.names')
        with open(names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    @property
    def wn_classes(self):
        """Category names."""
        names = os.path.join('./datasets/names/imagenetvid_wn.names')
        with open(names, 'r') as f:
            wn_classes = [line.strip() for line in f.readlines()]
        return wn_classes

    @property
    def image_ids(self):
        return [int(img_id[2][-15:-7])+int(img_id[2][-6:]) for img_id in self._items]

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
            ne_lf = os.path.join(root, 'ImageSets', 'VID', split + '_nonempty.txt')
            lf = os.path.join(root, 'ImageSets', 'VID', split + '.txt')
            if os.path.exists(ne_lf) and not self._allow_empty:
                lf = ne_lf

            print("Loading splits from: {}".format(lf))
            with open(lf, 'r') as f:
                ids_ = [(root, split, line.split()[0]) for line in f.readlines()]

            if not os.path.exists(ne_lf):
                ids_, str_ = self._verify_nonempty_annotations(ids_)  # ensure non-empty for this split
                print("Writing out new splits file: {}\n\n{}".format(ne_lf, str_))
                with open(ne_lf, 'w') as f:
                    for l in ids_:
                        f.write(l[2]+"\n")
                with open(os.path.join(root, 'ImageSets', 'VID', split + '_nonempty_stats.txt'), 'a') as f:
                    f.write(str_)

            ids += ids_

        if self._percent < 1:
            ids = [ids[i] for i in range(0, len(ids), int(1/self._percent))]

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
            if cls_name not in self.wn_classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)  # we dont need to minus 1 here as the already 0 > h/w
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            try:
                xmin, ymin, xmax, ymax = self._validate_label(xmin, ymin, xmax, ymax, width, height, anno_path)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id])

        if self._allow_empty and len(label) < 1:
            label.append([-1, -1, -1, -1, -1])
        return np.array(label)

    @staticmethod
    def _validate_label(xmin, ymin, xmax, ymax, width, height, anno_path):
        """Validate labels."""
        if not 0 <= xmin < width or not 0 <= ymin < height or not xmin < xmax <= width or not ymin < ymax <= height:
            print("box: {} {} {} {} incompatable with img size {}x{} in {}.".format(xmin,
                                                                                    ymin,
                                                                                    xmax,
                                                                                    ymax,
                                                                                    width,
                                                                                    height,
                                                                                    anno_path))

            xmin = min(max(0, xmin), width - 1)
            ymin = min(max(0, ymin), height - 1)
            xmax = min(max(xmin + 1, xmax), width)
            ymax = min(max(ymin + 1, ymax), height)

            print("new box: {} {} {} {}.".format(xmin, ymin, xmax, ymax))
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)
        return xmin, ymin, xmax, ymax

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

    def image_size(self, id):
        if len(self._im_shapes) == 0:
            for idx in tqdm(range(len(self._items)), desc="populating im_shapes"):
                self._load_label(idx, items=self._items)
        return self._im_shapes[self.image_ids.index(id)]

    def stats(self):
        cls_boxes = []
        n_samples = len(self._items)
        n_boxes = [0]*len(self.classes)
        for idx in tqdm(range(len(self._items))):
            for box in self._load_label(idx):
                n_boxes[int(box[4])] += 1

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

    def build_coco_json(self):

        os.makedirs(os.path.dirname(self._coco_path), exist_ok=True)

        # handle categories
        categories = list()
        for ci, (cls, wn_cls) in enumerate(zip(self.classes, self.wn_classes)):
            categories.append({'id': ci, 'name': cls, 'wnid': wn_cls})

        # handle images and boxes
        images = list()
        done_imgs = set()
        annotations = list()
        for idx in range(len(self)):
            img_id = self._items[idx]
            filename = self._anno_path.format(*img_id)
            width, height = self._im_shapes[idx]

            img_id = self.image_ids[idx]
            if img_id not in done_imgs:
                done_imgs.add(img_id)
                images.append({'file_name': filename,
                               'width': int(width),
                               'height': int(height),
                               'id': img_id})

            for box in self._load_label(idx):
                xywh = [int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])]
                annotations.append({'image_id': img_id,
                                    'id': len(annotations),
                                    'bbox': xywh,
                                    'area': int(xywh[2] * xywh[3]),
                                    'category_id': int(box[4]),
                                    'iscrowd': 0})

        with open(self._coco_path, 'w') as f:
            json.dump({'images': images, 'annotations': annotations, 'categories': categories}, f)


if __name__ == '__main__':
    train_dataset = ImageNetVidDetection(
        root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'), splits=['train'], allow_empty=False, frames=True)
    val_dataset = ImageNetVidDetection(
        root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'), splits=['val'], allow_empty=False, frames=True)

    print(train_dataset)
    print(val_dataset)
