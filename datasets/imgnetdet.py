"""ImageNet DET object detection dataset."""

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


class ImageNetDetection(VisionDataset):
    """ImageNet DET object detection dataset."""

    def __init__(self, root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
                 splits=['train'], allow_empty=False, transform=None, index_map=None, inference=False):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/ImageNetDET/ILSVRC')
            splits (list): a list of splits as strings (default is ['train'])
            allow_empty (bool): include samples that don't have any labelled boxes? (default is False)
            transform: the transform to apply to the image/video and label (default is None)
            index_map (dict): custom class to id dictionary (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(ImageNetDetection, self).__init__(root)
        self._im_shapes = {}
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._allow_empty = allow_empty
        self._inference = inference
        
        # setup a few paths
        self._coco_path = os.path.join(self.root, 'jsons', '_'.join([s for s in self._splits])+'.json')
        self._annotations_path = os.path.join('{}', 'Annotations', 'DET', '{}', '{}.xml')
        self._image_path = os.path.join('{}', 'Data', 'DET', '{}', '{}.JPEG')
        
        # setup the class index map
        self.index_map = index_map or dict(zip(self.wn_classes, range(self.num_class)))
        
        # load the samples
        self.samples = self._load_samples()
        
        # generate a sorted list of the sample ids
        self.sample_ids = sorted(list(self.samples.keys()))

        if not allow_empty:  # remove empty samples if desired
            self.samples, self.sample_ids = self._remove_empties()

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n'

    @property
    def classes(self):
        """
        Gets a list of class names as specified in the imagenetdet.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('./datasets/names/imagenetdet.names')
        with open(names_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    @property
    def wn_classes(self):
        """
        Gets a list of class names as specified in the imagenetdet_wn.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('./datasets/names/imagenetdet_wn.names')
        with open(names_file, 'r') as f:
            wn_classes = [line.strip() for line in f.readlines()]
        return wn_classes

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
        # load the image and label
        img_path = self.sample_path(idx)
        label = self._load_label(idx)
        img = mx.image.imread(img_path, 1)

        # do transform if desired
        if self._transform is not None:
            return self._transform(img, label)

        if self._inference:
            return img, label, idx
        else:
            return img, label

    def sample_path(self, idx):
        return self._image_path.format(*self.samples[self.sample_ids[idx]])

    def _load_samples(self):
        """
        Load the samples of this dataset using the settings supplied

        Returns:
            dict : a dict of image samples

        """
        ids = list()
        for split in self._splits:

            # load the splits file
            print("Loading splits from: {}".format(os.path.join(self.root, 'ImageSets', 'DET', split + '.txt')))
            with open(os.path.join(self.root, 'ImageSets', 'DET', split + '.txt'), 'r') as f:
                ids_ = [(self.root, split, line.split()[0]) for line in f.readlines()]

            ids += ids_

        samples = dict()
        for s in ids:
            assert s[-1] not in samples, logging.error("Sample keys not unique: {}".format(s[-1]))
            samples[s[-1]] = s
        return samples

    def _remove_empties(self):
        """
        removes empty samples from the set

        Returns:
            list: of the sample ids of non-empty samples
        """

        not_empty_file = os.path.join(self.root, 'ImageSets', 'DET', self._splits[0] + '_nonempty.txt')
        not_empty_stats_file = os.path.join(self.root, 'ImageSets', 'DET', self._splits[0] + '_nonempty_stats.txt')

        if os.path.exists(not_empty_file):  # if the splits file exists
            logging.info("Loading splits from: {}".format(not_empty_file))
            with open(not_empty_file, 'r') as f:
                good_sample_ids = [line.rstrip() for line in f.readlines()]

        else:  # if the splits file doesn't exist, make one
            good_sample_ids = list()
            removed = 0
            n_boxes = 0
            for idx in tqdm(range(len(self.sample_ids)), desc="Removing images that have 0 boxes"):
                n_boxes_in_sample = len(self._load_label(idx))
                if n_boxes_in_sample < 1:
                    removed += 1
                else:
                    n_boxes += n_boxes_in_sample
                    good_sample_ids.append(self.sample_ids[idx])

            str_ = "Removed {} out of {} images, leaving {} with {} boxes over {} classes.\n".format(
                removed, len(self.sample_ids), len(good_sample_ids), n_boxes, len(self.classes))

            logging.info("Writing out new splits file: {}\n\n{}".format(not_empty_file, str_))
            with open(not_empty_file, 'w') as f:
                for sample_id in good_sample_ids:
                    f.write('{}\n'.format(sample_id))
            with open(not_empty_stats_file, 'w') as f:
                f.write(str_)

        # remake the samples dict
        good_samples = dict()
        for sid in good_sample_ids:
            good_samples[sid] = self.samples[sid]

        return good_samples, good_sample_ids

    def _load_label(self, idx):
        """
        Parse the xml annotation files for a sample

        Args:
            idx (int): the sample index

        Returns:
            numpy.ndarray : labels of shape (n, 5) - [[xmin, ymin, xmax, ymax, cls_id], ...]
        """
        sample_id = self.sample_ids[idx]
        anno_path = self._annotations_path.format(*self.samples[sample_id])
        if not os.path.exists(anno_path):
            return np.array([[-1, -1, -1, -1, -1]])
        root = et.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if sample_id not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[sample_id] = (width, height)
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

            xmin, ymin, xmax, ymax = self._validate_label(xmin, ymin, xmax, ymax, width, height, anno_path)

            label.append([xmin, ymin, xmax, ymax, cls_id])

        if self._allow_empty and len(label) < 1:
            label.append([-1, -1, -1, -1, -1])
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

    def _verify_nonempty_annotations(self, sample_ids):
        """
        Checks annotations and returns only those that contain at least one box, as well as a lil stats str

        Args:
            sample_ids (list): a list of sample ids to process

        Returns:
            list: the sample ids kept in the set
            str: just a helpful string

        """
        good_sample_ids = []
        removed = 0
        n_boxes = 0
        for idx in tqdm(range(len(sample_ids)), desc="Removing images that have 0 boxes"):
            n_boxes_in_sample = len(self._load_label(idx))
            if n_boxes_in_sample < 1:
                removed += 1
            else:
                n_boxes += n_boxes_in_sample
                good_sample_ids.append(sample_ids[idx])

        str_ = "Removed {} out of {} images, leaving {} with {} boxes over {} classes.\n".format(
            removed, len(sample_ids), len(good_sample_ids), n_boxes, len(self.classes))

        return good_sample_ids, str_

    def image_size(self, id):
        if len(self._im_shapes) == 0:
            for idx in tqdm(range(len(self.samples)), desc="populating im_shapes"):
                self._load_label(idx)
        return self._im_shapes[self.image_ids.index(id)]

    def stats(self):
        """
        Get the dataset statistics

        Returns:
            str: an output string with all of the information
            list: a list of counts of the number of boxes per class

        """
        cls_boxes = []
        n_samples = len(self.samples)
        n_boxes = [0]*len(self.classes)
        for idx in tqdm(range(len(self.samples)), desc="Calculate stats for dataset"):
            for box in self._load_label(idx):
                n_boxes[int(box[4])] += 1

        out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n'.format('Split:', ', '.join(self._splits),
                                                                                    'Images:', n_samples,
                                                                                    'Boxes:', sum(n_boxes),
                                                                                    'Classes:', len(self.classes))
        out_str += '-'*45 + '\n'
        for i in range(len(n_boxes)):
            out_str += '{0: <3} {1: <10} {2: <25} {3}\n'.format(i, self.wn_classes[i], self.classes[i], n_boxes[i])
            cls_boxes.append([i, self.wn_classes[i], self.classes[i], n_boxes[i]])
        out_str += '-'*45 + '\n'

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
    train_dataset = ImageNetDetection(splits=['train'], allow_empty=False)
    for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
        pass
    print(train_dataset)

    val_dataset = ImageNetDetection(splits=['val'], allow_empty=False)
    for s in tqdm(val_dataset, desc='Test Pass of Validation Set'):
        pass
    print(val_dataset)
