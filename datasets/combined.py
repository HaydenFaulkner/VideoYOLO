"""Combined object detection dataset - allows you to combine datasets into one bigger one"""

from gluoncv.data.base import VisionDataset
import os

from nltk.corpus import wordnet as wn


def id_to_name(id):
    return wn.synset_from_pos_and_offset('n', int(id[1:]))._name


class CombinedDetection(VisionDataset):
    """Combined detection Dataset."""

    def __init__(self, datasets, root=os.path.join('datasets', 'combined'), class_tree=False):
        """
        :param datasets: list of dataset objects
        :param root: root path to store the dataset, str, default '/datasets/combined/'
        :param index_map: We can customize the class mapping by providing a str to int dict specifying how to map class
        names to indices. dict, default None
        """
        os.makedirs(root, exist_ok=True)
        super(CombinedDetection, self).__init__(root)

        self._datasets = datasets

        self._root = os.path.expanduser(root)
        self._class_tree = class_tree
        self._samples = self._load_samples()
        _, _, self._dataset_class_map = self._get_classes()

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
            lines = [l.rstrip().split for l in lines]
            parents = dict()
            for cls in lines:
                classes_wn.append(cls[0])
                classes.append(id_to_name(cls[0]))
                parents[cls[0]] = cls[1]

            for dataset_idx, dataset in enumerate(self._datasets):
                dataset_class_map = list()
                for wn_cls in dataset.wn_classes:
                    if wn_cls not in classes_wn:
                        classes_wn.append(wn_cls)
                        classes.append(cls)

                    dataset_class_map.append(classes_wn.index(wn_cls))
                dataset_class_maps.append(dataset_class_map)
        return classes, classes_wn, dataset_class_maps, parents

    @property
    def classes(self):
        """Category names."""
        return self._get_classes()[0]

    @property
    def wn_classes(self):
        """Category names."""
        return self._get_classes()[1]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        dataset_idx, dataset_sample_idx, idx = self._samples[idx]
        dataset = self._datasets[dataset_idx]

        # fix class id
        sample = dataset[dataset_sample_idx]
        for si in range(len(sample[1])):
            sample[1][si][4] = float(self._dataset_class_map[dataset_idx][int(sample[1][si][4])])
        return sample

    def _load_samples(self):
        samples = []
        for dataset_idx, dataset in enumerate(self._datasets):
            for idx in range(len(dataset)):
                samples.append((dataset_idx, idx, len(samples)))
        return samples

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

    # def build_coco_json(self):
    #
    #     os.makedirs(os.path.dirname(self._coco_path), exist_ok=True)
    #
    #     # handle categories
    #     categories = list()
    #     for ci, (cls, wn_cls) in enumerate(zip(self.classes, self.wn_classes)):
    #         categories.append({'id': ci, 'name': cls, 'wnid': wn_cls})
    #
    #     # handle images and boxes
    #     images = list()
    #     done_imgs = set()
    #     annotations = list()
    #     for idx in range(len(self)):
    #         dataset, dataset_idx, id, dataset_id = self._items[idx]
    #         img_id = dataset._items[dataset_idx]
    #         filename = dataset._anno_path.format(*img_id)
    #         width, height = dataset._im_shapes[dataset_idx]
    #
    #         img_id = self.image_ids[idx]
    #         if img_id not in done_imgs:
    #             done_imgs.add(img_id)
    #             images.append({'file_name': filename,
    #                            'width': int(width),
    #                            'height': int(height),
    #                            'id': img_id})
    #
    #         for box in self._load_label(idx):
    #             xywh = [int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])]
    #             annotations.append({'image_id': img_id,
    #                                 'id': len(annotations),
    #                                 'bbox': xywh,
    #                                 'area': int(xywh[2] * xywh[3]),
    #                                 'category_id': int(box[4]),
    #                                 'iscrowd': 0})
    #
    #     with open(self._coco_path, 'w') as f:
    #         json.dump({'images': images, 'annotations': annotations, 'categories': categories}, f)
    #
    #     return self._coco_path


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

    cd = CombinedDetection(datasets)

    print(cd.stats()[0])

    # for s in cd:
    #     print(s)
