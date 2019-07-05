"""Combined object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import json
from tqdm import tqdm
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from gluoncv.data.base import VisionDataset

from datasets.pascalvoc import VOCDetection

# todo accessing protected members of the dataset classes... what to do about this?


class CombinedDetection(VisionDataset):
    """Combined detection Dataset.
    Parameters
    ----------
    datasets : list of dataset objects
    root : str, default '~/mxnet/datasets/comb/'
        Path to folder storing the dataset.
    index_map : dict, default None
        We can customize the class mapping by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders of class labels.
    """

    def __init__(self, datasets, root=os.path.join('~', '.mxnet', 'datasets', 'comb'), index_map=None):
        os.makedirs(root, exist_ok=True)
        super(CombinedDetection, self).__init__(root)
        self._datasets = datasets

        # self._im_shapes = {}
        self._root = os.path.expanduser(root)
        # self._transform = transform
        self._items = self._load_items()
        self._coco_path = os.path.join(self._root, 'jsons', '_'.join([d.__class__.__name__ for d in self._datasets])+'.json')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats()[0] + '\n'

    def _get_classes(self):
        unique_classes = []
        unique_wn_classes = []
        for dataset in self._datasets:
            for cls, wn_cls in zip(dataset.classes, dataset.wn_classes):
                if wn_cls not in unique_wn_classes:
                    unique_wn_classes.append(wn_cls)
                    unique_classes.append(cls)
        return unique_classes, unique_wn_classes

    @property
    def classes(self):
        """Category names."""
        return self._get_classes()[0]

    @property
    def wn_classes(self):
        """Category names."""
        return self._get_classes()[1]

    @property
    def image_ids(self):
        return [item[2] for item in self._items]  # ids here are just 0->len(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        dataset, dataset_idx, id, dataset_id = self._items[idx]

        img_id = dataset._items[dataset_idx]
        img_path = dataset._image_path.format(*img_id)
        label = dataset._label_cache[idx] if dataset._label_cache else dataset._load_label(dataset_idx)
        img = mx.image.imread(img_path, 1)
        if dataset._transform is not None:
            return dataset._transform(img, label)
        return img, label

    def _load_items(self):
        ids = []
        for dataset in self._datasets:
            for idx, item in enumerate(dataset._items):
                ids.append((dataset, idx, len(ids), dataset.image_ids[idx]))
        return ids

    def _load_label(self, idx):
        dataset, dataset_idx, id, dataset_id = self._items[idx]
        return dataset._load_label(dataset_idx)

    def image_size(self, id):
        idx = self.image_ids.index(id)  # should technically be the same...
        dataset, dataset_idx, id, dataset_id = self._items[idx]
        return dataset._im_shapes[dataset_idx]

    def stats(self):
        cls_boxes = []
        n_samples = len(self._items)
        n_boxes = [0] * len(self.classes)
        for idx in tqdm(range(len(self._items))):
            for box in self._load_label(idx):
                n_boxes[int(box[4])] += 1

        out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n'.format(
            'Datasets: Splits', '\n'.join([d.__class__.__name__ + ': '+', '.join(
                [str(s[0]) + s[1] for s in d._splits]) for d in self._datasets]),
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
            dataset, dataset_idx, id, dataset_id = self._items[idx]
            img_id = dataset._items[dataset_idx]
            filename = dataset._anno_path.format(*img_id)
            width, height = dataset._im_shapes[dataset_idx]

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

        return self._coco_path


if __name__ == '__main__':
    train_dataset = VOCDetection(
        root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'), splits=[(2007, 'trainval'), (2012, 'trainval')])
    val_dataset = VOCDetection(
        root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'), splits=[(2007, 'test')])
    comb_dataset = CombinedDetection(datasets=[train_dataset, val_dataset],
                                     root=os.path.join('datasets', 'Combined'))

    print(comb_dataset)
