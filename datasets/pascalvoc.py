"""Pascal VOC object detection dataset - Edited  from gluoncv
"""
from absl import app, flags, logging
from gluoncv.data.base import VisionDataset
import json
import mxnet as mx
import numpy as np
import os
from tqdm import tqdm
try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et


class VOCDetection(VisionDataset):
    """Pascal VOC object detection dataset."""

    def __init__(self, root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
                 splits=[(2007, 'trainval'), (2012, 'trainval')],
                 transform=None, index_map=None, preload_label=True, difficult=True, inference=False):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/PascalVOC/VOCdevkit')
            splits (list): a list of splits as tuples (default is [(2007, 'trainval'), (2012, 'trainval')])
            transform: the transform to apply to the image/video and label (default is None)
            index_map (dict): custom class to id dictionary (default is None)
            preload_label (bool): load and store all labels in system memory (default is True)
            difficult (bool): include difficult samples (default is True) 
            inference (bool): are we doing inference? (default is False)
        """
        super(VOCDetection, self).__init__(root)
        self._im_shapes = {}
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._difficult = difficult
        self._inference = inference

        # setup a few paths
        self._coco_path = os.path.join(self.root, 'jsons', '_'.join([str(s[0]) + s[1] for s in self._splits])+'.json')
        self._annotations_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')

        # setup the class index map
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))

        # load the samples
        self.samples = self._load_samples()
        
        # load the labels into memory
        self._labels = self._preload_labels() if preload_label else None

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats()[0] + '\n'

    @property
    def classes(self):
        """
        Gets a list of class names as specified in the pascalvoc.names file

        Returns:
            list : a list of strings

        """
        names = os.path.join('./datasets/names/pascalvoc.names')
        with open(names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    @property
    def wn_classes(self):
        """
        Gets a list of class names as specified in the pascalvoc_wn.names file

        Returns:
            list : a list of strings

        """
        names = os.path.join('./datasets/names/pascalvoc_wn.names')
        with open(names, 'r') as f:
            wn_classes = [line.strip() for line in f.readlines()]
        return wn_classes

    # @property
    # def image_ids(self):
    #     return [int(img_id[1]) for img_id in self.samples]

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
        img_path = self._image_path.format(*self.samples[self.sample_ids[idx]])
        label = self._labels[idx] if self._labels else self._load_label(idx)
        img = mx.image.imread(img_path, 1)

        if self._transform is not None:
            return self._transform(img, label)

        if self._inference:
            return img, label, idx
        else:
            return img, label

    # def sample_path(self, idx):
    #     img_id = self.samples[idx]
    #     img_path = self._image_path.format(*img_id)
    #     return img_path

    def _load_samples(self):
        """
        Load the samples of this dataset using the settings supplied

        Returns:
            dict : a dict of image samples

        """
        ids = list()
        for year, name in self._splits:
            root = os.path.join(self.root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]

        samples = dict()
        for s in ids:
            assert s[-1] not in samples, logging.error("Sample keys not unique: {}".format(s[-1]))
            samples[s[-1]] = s
        return samples

    def _load_label(self, idx):
        """
        Parse the xml annotation files for a sample

        Args:
            idx (int): the sample index

        Returns:
            numpy.ndarray : labels of shape (n, 6) - [[xmin, ymin, xmax, ymax, cls_id, difficult], ...]
        """
        sample_id = self.sample_ids[idx]
        anno_path = self._annotations_path.format(*self.samples[sample_id])
        if not os.path.exists(anno_path):
            return np.array([[-1, -1, -1, -1, -1, -1]])
        root = et.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if sample_id not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[sample_id] = (width, height)
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
            if self._difficult:
                label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
            else:
                label.append([xmin, ymin, xmax, ymax, cls_id, 0])

        if len(label) < 1:
            label.append([-1, -1, -1, -1, -1, -1])
        return np.array(label)

    @staticmethod
    def _validate_label(xmin, ymin, xmax, ymax, width, height, anno_path):
        """Validate labels."""
        if not 0 <= xmin < width or not 0 <= ymin < height or not xmin < xmax <= width or not ymin < ymax <= height:
            logging.warning("box: {} {} {} {} incompatible with img size {}x{} in {}.".format(xmin, ymin, xmax, ymax,
                                                                                              width, height, anno_path))

            xmin = min(max(0, xmin), width - 1)
            ymin = min(max(0, ymin), height - 1)
            xmax = min(max(xmin + 1, xmax), width)
            ymax = min(max(ymin + 1, ymax), height)

            logging.info("new box: {} {} {} {}.".format(xmin, ymin, xmax, ymax))
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _validate_class_names(class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            logging.warning('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.info("Preloading labels into memory...")
        return [self._load_label(idx) for idx in range(len(self))]

    # def image_size(self, id):
    #     return self._im_shapes[self.image_ids.index(id)]

    def stats(self):
        """
        Get the dataset statistics

        Returns:
            str: an output string with all of the information
            list: a list of counts of the number of boxes per class

        """
        cls_boxes = []
        n_samples = len(self._labels)
        n_boxes = [0] * len(self.classes)
        for label in self._labels:
            for box in label:
                n_boxes[int(box[4])] += 1

        out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n'.format('Split:', ', '.join([str(s[0]) + s[1] for s in self._splits]),
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
        """
        Builds a groundtruth ms coco style .json for evaluation on this dataset

        Returns:
            str: the path to the output .json file

        """

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
            sample_id = self.sample_ids[idx]
            filename = self._annotations_path.format(*self.samples[sample_id])
            width, height = self._im_shapes[sample_id]

            if sample_id not in done_imgs:
                done_imgs.add(sample_id)
                images.append({'file_name': filename,
                               'width': int(width),
                               'height': int(height),
                               'id': sample_id})

            for box in self._load_label(idx):
                xywh = [int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])]
                annotations.append({'image_id': sample_id,
                                    'id': len(annotations),
                                    'bbox': xywh,
                                    'area': int(xywh[2] * xywh[3]),
                                    'category_id': int(box[4]),
                                    'iscrowd': 0})

        with open(self._coco_path, 'w') as f:
            json.dump({'images': images, 'annotations': annotations, 'categories': categories}, f)

        return self._coco_path


if __name__ == '__main__':
    train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
    for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
        pass
    print(train_dataset)

    val_dataset = VOCDetection(splits=[(2007, 'test')])
    for s in tqdm(val_dataset, desc='Test Pass of Validation Set'):
        pass
    print(val_dataset)
