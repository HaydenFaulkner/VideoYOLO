"""Combined object detection dataset - allows you to combine datasets into one bigger one"""

from gluoncv.data.base import VisionDataset
import json
import os
import mxnet as mx
import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def id_to_name(id):
    return wn.synset_from_pos_and_offset('n', int(id[1:]))._name


class CombinedDetection(VisionDataset):
    """Combined detection Dataset."""

    def __init__(self, datasets, root=os.path.join('datasets', 'combined'), class_tree=False, validation=False, inference=False):
        """
        :param datasets: list of dataset objects
        :param root: root path to store the dataset, str, default '/datasets/combined/'
        :param index_map: We can customize the class mapping by providing a str to int dict specifying how to map class
        names to indices. dict, default None
        """
        os.makedirs(root, exist_ok=True)
        super(CombinedDetection, self).__init__(root)

        self._datasets = datasets
        self.name = 'comb'
        self._root = os.path.expanduser(root)
        self._class_tree = class_tree
        self._validation = validation
        self._inference = inference
        self._samples = self._load_samples()
        self.sample_ids = list(self._samples.keys())
        self._classes, self.wn_classes, self._dataset_class_map, self.parents = self._get_classes()

        self.class_levels = self.get_levels()
        self.leaves = self.get_leaves()

        self._coco_path = os.path.join(self._root, 'jsons', '_'.join([d.name for d in datasets])+'.json')

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats()[0] + '\n'

    def _get_classes(self):
        """
        Take the classes from each dataset and reassign for the combined dataset
        :return:
        """
        classes = list()
        classes_wn = list()
        dataset_class_maps = list()
        parents = None
        if self._class_tree:
            with open(os.path.join('datasets', 'trees', 'filtered_det.tree'), 'r') as f:
                lines = f.readlines()
            lines = [l.rstrip().split() for l in lines]
            parents = dict()
            for cls in lines:
                classes_wn.append(cls[0])
                classes.append(id_to_name(cls[0]))
                parents[cls[0]] = cls[1]

            # handle swapping of ids
            with open(os.path.join('datasets', 'trees', 'new_classes.txt'), 'r') as f:
                lines = f.readlines()
            lines = [l.rstrip().split() for l in lines]
            swap_ids = dict()
            for ids in lines:
                swap_ids[ids[0]] = ids[1]

        for dataset_idx, dataset in enumerate(self._datasets):
            dataset_class_map = list()
            for cls in dataset.wn_classes:
                if cls not in classes_wn:
                    if self._class_tree:  # take into account where a swap needs to be done to the id
                        assert cls in swap_ids, '%s not in swap_ids, should be added to new_classes.txt' % cls
                        cls = swap_ids[cls]
                    else:
                        classes_wn.append(cls)
                        classes.append(id_to_name(cls))

                dataset_class_map.append(classes_wn.index(cls))
            dataset_class_maps.append(dataset_class_map)
        return classes, classes_wn, dataset_class_maps, parents

    @property
    def classes(self):
        return self._classes

    def get_sample_ids(self):
        return self.sample_ids

    def get_levels(self):
        levels = list()
        for c in self.wn_classes:
            lvl = 0
            p = c
            while p != 'ROOT':
                p = self.parents[p]
                lvl += 1
            levels.append(lvl)
        return levels

    def get_leaves(self):
        is_parent = set()

        for c in self.wn_classes:
            is_parent.add(self.parents[c])

        leaves = list()
        for c in self.wn_classes:
            if c in is_parent:
                leaves.append(0)
            else:
                leaves.append(1)

        return leaves

    def on_branch(self, c1, c2):
        """Are these two classes on the same lineage/branch"""
        if c1 == c2:
            return True
        child = self.wn_classes[max(c1, c2)]
        parent = self.wn_classes[min(c1, c2)]

        p = child
        while p != 'ROOT':
            p = self.parents[p]
            if p == parent:
                return True
        return False

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        dataset_idx, dataset_sample_idx = self._samples[idx]
        dataset = self._datasets[dataset_idx]

        # fix class id
        sample = list(dataset[dataset_sample_idx])
        if self._class_tree and self._validation:  # pass out y = [l,t,r,b,c] for all c's repeating the boxes for each c
            dup_boxes = list()
            for bi in range(len(sample[1])):
                cls = int(self._dataset_class_map[dataset_idx][int(sample[1][bi][4])])
                if cls < 0:
                    continue

                bx = np.copy(sample[1][bi])
                bx[4] = cls
                dup_boxes.append(bx)
                while self.wn_classes[cls] in self.parents:
                    if self.parents[self.wn_classes[cls]] == 'ROOT':
                        break
                    cls = self.wn_classes.index(self.parents[self.wn_classes[cls]])
                    bx = np.copy(sample[1][bi])
                    bx[4] = cls
                    dup_boxes.append(bx)
            sample[1] = np.vstack(dup_boxes)

        elif self._class_tree:  # pass out y=[l,t,r,b,c1,c2,c3,c4,....] with binary digits per class
            boxes = mx.nd.zeros((sample[1].shape[0], 4 + len(self.classes)))
            boxes[:, :4] = sample[1][:, :4]

            for bi in range(len(sample[1])):
                cls = int(self._dataset_class_map[dataset_idx][int(sample[1][bi][4])])
                if cls < 0:
                    boxes[bi, :] = -1
                    continue
                clss = [cls+4]
                while self.wn_classes[cls] in self.parents:
                    if self.parents[self.wn_classes[cls]] == 'ROOT': break
                    cls = self.wn_classes.index(self.parents[self.wn_classes[cls]])
                    clss.append(cls+4)
                clss.reverse()
                boxes[bi, clss] = 1
            sample[1] = boxes.asnumpy()
        else:  # y = a single [l,t,r,b,c] with the updated class
            for bi in range(len(sample[1])):
                sample[1][bi][4] = float(self._dataset_class_map[dataset_idx][int(sample[1][bi][4])])

        if self._inference:
            return sample[0], sample[1], idx
        return sample[0], sample[1]

    def _load_samples(self):
        samples = dict()
        for dataset_idx, dataset in enumerate(self._datasets):
            for idx in range(len(dataset)):
                samples[len(samples)] = (dataset_idx, idx)
        return samples

    def im_shapes(self, sid):
        dataset_idx, dataset_sample_idx = self._samples[sid]
        dataset = self._datasets[dataset_idx]
        return dataset.im_shapes(dataset.sample_ids[dataset_sample_idx])
    
    def _load_label(self, idx):
        dataset_idx, dataset_sample_idx = self._samples[self.sample_ids[idx]]
        dataset = self._datasets[dataset_idx]
        return dataset._load_label(dataset_sample_idx)

    def sample_path(self, idx):
        dataset_idx, dataset_sample_idx = self._samples[self.sample_ids[idx]]
        dataset = self._datasets[dataset_idx]
        return dataset.sample_path(dataset_sample_idx)

    def stats(self):
        cls_boxes = []
        n_samples = len(self._samples)
        n_boxes = [0] * len(self.classes)
        for _, l in self:
            for box in l:
                n_boxes[int(box[4])] += 1

        out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n'.format(
            'Images:', n_samples,
            'Boxes:', sum(n_boxes),
            'Classes:', len(self.classes))
        out_str += '-'*35 + '\n'
        for i in range(len(n_boxes)):
            out_str += '{0: <3} {1: <10} {2: <20} {3}\n'.format(i, self.wn_classes[i], self.classes[i], n_boxes[i])
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
        for idx in tqdm(range(len(self)), desc='generating coco eval'):
            sample_id = self.sample_ids[idx]
            filename = self.sample_path(idx)
            width, height = self.im_shapes(sid=sample_id)

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

    from datasets.pascalvoc import VOCDetection
    from datasets.mscoco import COCODetection
    from datasets.imgnetdet import ImageNetDetection
    from datasets.imgnetvid import ImageNetVidDetection

    datasets = list()
    datasets.append(VOCDetection(splits=[(2007, 'test')]))
    print('Loaded VOC')
    datasets.append(COCODetection(splits=['instances_val2017'], allow_empty=True))
    print('Loaded COCO')
    datasets.append(ImageNetDetection(splits=['val'], allow_empty=True))
    print('Loaded DET')
    datasets.append(ImageNetVidDetection(splits=[(2017, 'val')], allow_empty=True, every=25, window=[1, 1]))
    print('Loaded VID')

    cd = CombinedDetection(datasets, class_tree=True, validation=True)

    cd.build_coco_json()
    # from tqdm import tqdm
    # for s in tqdm(cd):
    #     pass
