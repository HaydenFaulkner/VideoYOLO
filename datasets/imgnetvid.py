"""ImageNet VID object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import json
import math
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
                 splits=((2017, 'train'),), allow_empty=False, videos=False,
                 transform=None, index_map=None, frames=1, inference=False):
        super(ImageNetVidDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        assert len(splits) == 1, print('Can only take one split currently as otherwise conflicting image ids')
        self._splits = splits
        self._videos = videos
        self._frames = frames
        self._inference = inference
        self._allow_empty = allow_empty
        self._coco_path = os.path.join(self._root, 'jsons', '_'.join([str(s[0]) + s[1] for s in self._splits])+'.json')
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
        if self._videos:
            return [sample[3].split('/')[0] for sample in self._items] # todo check which we want?
            # return [id for id in [sample[4] for sample in self._items]] # todo check which we want?
        else:
            return [sample[0] for sample in self._items]

    @property
    def motion_ious(self):
        ious = self._load_motion_ious()
        motion_ious = np.array([v for k, v in ious.items() if int(k) in self.image_ids])
        return motion_ious

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        if not self._videos:
            img_id = self._items[idx]
            img_path = self._image_path.format(*img_id[1:])
            label = self._load_label(idx)[:, :-1]  # remove track id
            img = mx.image.imread(img_path, 1)

            if self._inference:
                if self._transform is not None:
                    return self._transform(img, label, idx)
                return img, label, idx
            else:
                if self._transform is not None:
                    return self._transform(img, label)
                return img, label
        else:
            sample_id = self._items[idx]
            vid = None
            labels = None
            for frame in sample_id[3]:
                img_id = (sample_id[0], sample_id[1], sample_id[2]+'/'+frame)
                img_path = self._image_path.format(*img_id)
                label = self._load_label(idx, frame=frame)
                img = mx.image.imread(img_path, 1)
                if self._transform is not None:
                    img, label = self._transform(img, label)

                label = self._pad_to_dense(label, 20)
                if labels is None:
                    # vid = mx.ndarray.expand_dims(img, 0)  # todo investigate why this uses so much mem, maybe long video?
                    labels = np.expand_dims(label, 0)
                else:
                    # vid = mx.ndarray.concatenate([vid, mx.ndarray.expand_dims(img, 0)])
                    labels = np.concatenate((labels, np.expand_dims(label, 0)))
            return None, labels

    def sample_path(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id[1:])
        return img_path

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, split in splits:
            root = self._root
            ne_lf = os.path.join(root, 'ImageSets', 'VID', split + '_nonempty.txt')
            lf = os.path.join(root, 'ImageSets', 'VID', split + '.txt')
            if os.path.exists(ne_lf) and not self._allow_empty:
                lf = ne_lf

            print("Loading splits from: {}".format(lf))
            with open(lf, 'r') as f:
                ids_ = [(int(line.split()[1]), root, split, line.split()[0]) for line in f.readlines()]

            if not os.path.exists(ne_lf) and not self._allow_empty:
                ids_, str_ = self._verify_nonempty_annotations(ids_)  # ensure non-empty for this split
                print("Writing out new splits file: {}\n\n{}".format(ne_lf, str_))
                with open(ne_lf, 'w') as f:
                    for l in ids_:
                        f.write('{} {}\n'.format(l[3], l[0]))
                with open(os.path.join(root, 'ImageSets', 'VID', split + '_nonempty_stats.txt'), 'a') as f:
                    f.write(str_)

            if year == 2015:
                ids_ = [id for id in ids_ if 'ILSVRC2015' in id[-1]]

            ids += ids_

        if self._frames == 1:
            return ids

        # We only want a subset of each video, so need to find the videos
        vid_ids = []
        past_vid_id = ''
        for id in ids:
            vid_id = id[3][:-7]
            frame = id[3][-6:]
            if vid_id != past_vid_id:
                if past_vid_id:
                    if self._frames < 1:  # cut down per video
                        frames = [frames[i] for i in range(0, len(frames), int(1/self._frames))]
                        sample_ids = [sample_ids[i] for i in range(0, len(sample_ids), int(1/self._frames))]
                    elif self._frames > 1:  # cut down per video
                        frames = [frames[i] for i in range(0, len(frames), int(math.ceil(len(frames)/self._frames)))]
                        sample_ids = [sample_ids[i] for i in range(0, len(sample_ids), int(math.ceil(len(frames)/self._frames)))]

                    vid_ids.append((past_id[1], past_id[2], past_vid_id, frames, sample_ids))
                past_id = id
                frames = [frame]
                sample_ids = [id[0]]
                past_vid_id = vid_id
            else:
                frames.append(frame)
                sample_ids.append(id[0])

        if self._frames < 1: # cut down per video
            frames = [frames[i] for i in range(0, len(frames), int(1 / self._frames))]
            sample_ids = [sample_ids[i] for i in range(0, len(sample_ids), int(1 / self._frames))]
        elif self._frames > 1:  # cut down per video
            frames = [frames[i] for i in range(0, len(frames), int(math.ceil(len(frames)/self._frames)))]
            sample_ids = [sample_ids[i] for i in range(0, len(sample_ids), int(math.ceil(len(frames)/self._frames)))]

        vid_ids.append((id[1], id[2], vid_id, frames, sample_ids))

        if self._videos:
            return vid_ids
        else:
            # reconstruct the ids from vid ids
            frame_ids = []
            for vid_id in vid_ids:
                for frame_name, frame_id in zip(vid_id[3], vid_id[4]):
                    frame_ids.append((frame_id, vid_id[0], vid_id[1], vid_id[2]+'/'+frame_name))
            return frame_ids

    def _load_label(self, idx, items=None, frame=None):
        """Parse xml file and return labels."""
        if items:  # used to process items before they are actually assigned to self
            img_id = items[idx]
        else:
            img_id = self._items[idx]

        anno_path = self._anno_path.format(*img_id[1:])
        if self._videos:
            assert frame is not None
            img_id = (img_id[0], img_id[1], img_id[2]+'/'+frame)
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
            trk_id = int(obj.find('trackid').text)
            xml_box = obj.find('bndbox')
            # assert int(obj.find('trackid').text) == len(label), print(anno_path)  # this ensures that boxes are ordered based on their track ids (useful for making the the motion iou)
            xmin = float(xml_box.find('xmin').text)  # we dont need to minus 1 here as the already 0 > h/w
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            try:
                xmin, ymin, xmax, ymax = self._validate_label(xmin, ymin, xmax, ymax, width, height, anno_path)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, trk_id])

        if self._allow_empty and len(label) < 1:
            label.append([-1, -1, -1, -1, -1, -1])
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

    def _pad_to_dense(self, labels, maxlen=100):
        """Appends the minimal required amount of zeroes at the end of each
         array in the jagged array `M`, such that `M` looses its jagedness."""

        x = -np.ones((maxlen, 6))
        for enu, row in enumerate(labels):
            x[enu, :] += row + 1
        return x

    def image_size(self, id):
        if len(self._im_shapes) == 0:
            for idx in tqdm(range(len(self._items)), desc="populating im_shapes"):
                self._load_label(idx, items=self._items)
        return self._im_shapes[self.image_ids.index(id)]

    def stats(self):
        cls_boxes = []
        n_samples = len(self._items)
        n_boxes = [0]*len(self.classes)
        n_instances = [0]*len(self.classes)
        past_vid_id = ''
        for idx in tqdm(range(len(self._items))):
            vid_id = self._items[idx][3][:-7]
            frame = self._items[idx][3][-6:]
            if vid_id != past_vid_id:
                vid_instances = []
                past_vid_id = vid_id
            for box in self._load_label(idx):
                n_boxes[int(box[4])] += 1
                if int(box[5]) not in vid_instances:
                    vid_instances.append(int(box[5]))
                    n_instances[int(box[4])] += 1

        out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n{8: <10} {9}\n'.format(
            'Split:', ', '.join([str(s[0]) + s[1] for s in self._splits]),
            'Images:', n_samples,
            'Boxes:', sum(n_boxes),
            'Instances:', sum(n_instances),
            'Classes:', len(self.classes))
        out_str += '-'*35 + '\n'
        for i in range(len(n_boxes)):
            out_str += '{0: <3} {1: <10} {2: <15} {3} {4}\n'.format(i, self.wn_classes[i], self.classes[i],
                                                                    n_boxes[i], n_instances[i])
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

    def _load_motion_ious(self):
        path = os.path.join(self._root, '{}_motion_ious.json'.format(self._splits[0][1]))
        if not os.path.exists(path):
            generate_motion_ious(self._root, self._splits[0][1])

        with open(path, 'r') as f:
            ious = json.load(f)

        return ious


def iou(bb, bbgt):
    """
    helper single box iou calculation function for generate_motion_ious
    """
    ov = 0
    iw = np.min((bb[2], bbgt[2])) - np.max((bb[0], bbgt[0])) + 1
    ih = np.min((bb[3], bbgt[3])) - np.max((bb[1], bbgt[1])) + 1
    if iw > 0 and ih > 0:
        # compute overlap as area of intersection / area of union
        intersect = iw * ih
        ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
             (bbgt[2] - bbgt[0] + 1.) * \
             (bbgt[3] - bbgt[1] + 1.) - intersect
        ov = intersect / ua
    return ov


def generate_motion_ious(root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'), split='val'):
    """
    Used to generate motion_ious .json files matching that of imagenet_vid_groundtruth_motion_iou.mat from FGFA
    Except these are keyed on the image ids listed beside each frame in the ImageSets

    Note This script is relatively intensive

    :param root: the root path of the imagenetvid set
    :param split: train or val, no year necessary
    """

    dataset = ImageNetVidDetection(root=root, splits=[(2017, split)], allow_empty=True, videos=True, percent=1)

    all_ious = []
    all_ious = {} # using dict for better removing of elements on loading based on sample_id
    sample_id = 1 # messy but works
    for idx in tqdm(range(len(dataset)), desc="Generation motion iou groundtruth"):
        _, video = dataset[idx]
        for frame in range(len(video)):
            frame_ious = []
            for box_idx in range(len(video[frame])):
                trk_id = video[frame][box_idx][5]
                if trk_id > -1:
                    ious = []
                    for i in range(-10, 11):
                        frame_c = frame + i
                        if 0 <= frame_c < len(video) and i != 0:
                            for c_box_idx in range(len(video[frame_c])):
                                c_trk_id = video[frame_c][c_box_idx][5]
                                if trk_id == c_trk_id:
                                    a = video[frame][box_idx]
                                    b = video[frame_c][c_box_idx]
                                    ious.append(iou(a, b))
                                    break

                    frame_ious.append(np.mean(ious))
            # if frame_ious:  # if this frame has no boxes we add a 0.0
            #     all_ious.append(frame_ious)
            # else:
            #     all_ious.append([0.0])
            if frame_ious:  # if this frame has no boxes we add a 0.0
                all_ious[sample_id] = frame_ious
            else:
                all_ious[sample_id] = [0.0]
            sample_id += 1

    with open(os.path.join(root, '{}_motion_ious.json'.format(split)), 'w') as f:
        json.dump(all_ious, f)


if __name__ == '__main__':
    train_dataset = ImageNetVidDetection(
        root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'), splits=[(2017, 'train')],
        allow_empty=False, videos=False, frames=0.04)
    val_dataset = ImageNetVidDetection(
        root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'), splits=[(2017, 'val')],
        allow_empty=False, videos=False, frames=0.04)
    test_dataset = ImageNetVidDetection(
        root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'), splits=[(2015, 'test')],
        allow_empty=True, videos=False)

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)