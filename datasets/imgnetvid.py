"""ImageNet VID object detection dataset."""

from absl import app, flags, logging
from gluoncv.data.base import VisionDataset
import json
import math
import mxnet as mx
import numpy as np
import os
from tqdm import tqdm
try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et


class ImageNetVidDetection(VisionDataset):
    """ImageNet VID object detection dataset."""

    def __init__(self, root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
                 splits=[(2017, 'train')], allow_empty=True, videos=False,
                 transform=None, index_map=None, frames=1, inference=False,
                 window_size=1, window_step=1):

        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/ImageNetVID/ILSVRC')
            splits (list): a list of splits as tuples (default is [(2017, 'train')])
            allow_empty (bool): include samples that don't have any labelled boxes? (default is True)
            videos (bool): interpret samples as full videos rather than frames (default is False)
            transform: the transform to apply to the image/video and label (default is None)
            index_map (dict): custom class to id dictionary (default is None)
            frames (float): Based per video - and is evenly sampled, NOT randomly sampled (default is 1)
                            <1: Percent of the full dataset to take -> range(0, len(video), int(1/frames))
                                eg. .04 takes every 25th frame
                            >1: This many frames per video -> range(0, len(video), int(ceil(len(video)/frames)))
                            =1: Every sample used -> full dataset
            inference (bool): are we doing inference? (default is False)
            window_size (int): how many frames does a sample consist of? ie. the temporal window size (default is 1)
            window_step (int): the step distance of the temporal window (default is 1)
        """
        super(ImageNetVidDetection, self).__init__(root)
        self._im_shapes = {}
        self.root = os.path.expanduser(root)
        self._transform = transform
        assert len(splits) == 1, logging.error('Can only take one split currently as otherwise conflicting image ids')
        self._splits = splits
        self._videos = videos
        self._frames = frames
        self._inference = inference
        if videos or window_size > 1:
            allow_empty = True  # allow true if getting video volumes, prevent empties when doing framewise only
        self._allow_empty = allow_empty
        self._window_size = window_size
        self._window_step = window_step
        self._windows = None
        
        # setup a few paths
        self._coco_path = os.path.join(self.root, 'jsons', '_'.join([str(s[0]) + s[1] for s in self._splits])+'.json')
        self._annotations_path = os.path.join('{}', 'Annotations', 'VID', '{}', '{}.xml')
        self._image_path = os.path.join('{}', 'Data', 'VID', '{}', '{}.JPEG')
        
        # setup the class index map
        self.index_map = index_map or dict(zip(self.wn_classes, range(self.num_class)))
        
        # load the samples
        self.samples = self._load_samples()
        
        # generate a sorted list of the sample ids
        self.sample_ids = sorted(list(self.samples.keys()))

        if not allow_empty:  # remove empty samples if desired
            self.samples, self.sample_ids = self._remove_empties()
        print()

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats()[0] + '\n'

    @property
    def classes(self):
        """
        Gets a list of class names as specified in the imagenetvid.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('datasets', 'names', 'imagenetvid.names')
        with open(names_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    @property
    def wn_classes(self):
        """
        Gets a list of class names as specified in the imagenetvid_wn.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('datasets', 'names', 'imagenetvid_wn.names')
        with open(names_file, 'r') as f:
            wn_classes = [line.strip() for line in f.readlines()]
        return wn_classes

    @property
    def motion_ious(self):
        path = os.path.join(self.root, '{}_motion_ious.json'.format(self._splits[0][1]))
        if not os.path.exists(path):
            generate_motion_ious()

        with open(path, 'r') as f:
            motion_ious = json.load(f)
        
        # filter only the samples in the set
        motion_ious = np.array([v for k, v in motion_ious.items() if int(k) in self.sample_ids])
        
        return motion_ious

    def __len__(self):
        return len(self.sample_ids)

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
        if not self._videos:  # frames are samples
            img_path = self.sample_path(idx)
            label = self._load_label(idx)[:, :-1]  # remove track id

            if self._window_size > 1:  # lets load the temporal window
                imgs = None
                window_sample_ids = self._windows[self.sample_ids[idx]]

                # go through the sample ids for the window
                for sid in window_sample_ids:
                    img_path = self._image_path.format(*self.samples[sid])
                    img = mx.image.imread(img_path)

                    if self._transform is not None:  # transform each image in the window
                        img, _ = self._transform(img, label)  # todo check we transform the same (NOT rand acr win)
                        # todo i think we should change the transforms to handle video volumes instead of per img here
                    if imgs is None:
                        # imgs = img  # is the first frame in the window
                        imgs = mx.ndarray.expand_dims(img, axis=0)  # is the first frame in the window
                    else:
                        # imgs = mx.ndarray.concatenate([imgs, img], axis=2)  # isn't first frame, concat to the window
                        imgs = mx.ndarray.concatenate([imgs, mx.ndarray.expand_dims(img, axis=0)], axis=0)
                img = imgs

                # transform the label
                if self._transform is not None:
                    _, label = self._transform(img, label)

                # necessary to prevent asynchronous operation overload and memory issue
                # https://discuss.mxnet.io/t/memory-leak-when-running-cpu-inference/3256
                mx.nd.waitall()

            else:  # window size is 1, so just load one image
                img = mx.image.imread(img_path, 1)
                if self._transform is not None:
                    img, label = self._transform(img, label)

            if self._inference:  # in inference we want to return the idx also
                return img, label, idx
            else:
                return img, label
        else:
            sample_id = self.sample_ids[idx]
            sample = self.samples[sample_id]
            vid = None
            labels = None
            for frame_id in sample[3]:  # for each frame in the video
                # load the frame and the label
                img_id = (sample[0], sample[1], os.path.join(sample[2], frame_id))
                img_path = self._image_path.format(*img_id)
                label = self._load_label(idx, frame_id=frame_id)
                img = mx.image.imread(img_path, 1)

                # transform the image and label
                if self._transform is not None:
                    img, label = self._transform(img, label)

                # pad label to ensure all same size for concatenation
                label = self._pad_to_dense(label, 20)

                # concatenate the data to make video and label volumes
                if labels is None:
                    vid = mx.ndarray.expand_dims(img, axis=0)
                    labels = np.expand_dims(label, axis=0)
                else:
                    vid = mx.ndarray.concatenate([vid, mx.ndarray.expand_dims(img, axis=0)], axis=0)
                    labels = np.concatenate((labels, np.expand_dims(label, axis=0)), axis=0)

            # necessary to prevent asynchronous operation overload and memory issue
            # https://discuss.mxnet.io/t/memory-leak-when-running-cpu-inference/3256
            mx.nd.waitall()

            if self._inference:  # in inference we want to return the idx also
                return vid, labels, idx
            else:
                return vid, labels

    def sample_path(self, idx):
        if self._videos:
            sample = self.samples[self.sample_ids[idx]]
            return os.path.join(sample[0], sample[1], sample[2])

        return self._image_path.format(*self.samples[self.sample_ids[idx]])

    def _remove_empties(self):
        """
        removes empty samples from the set

        Returns:
            list: of the sample ids of non-empty samples
        """

        assert not self._videos, logging.error("Can't exclude non-empty samples for videos")

        not_empty_file = os.path.join(self.root, 'ImageSets', 'VID', self._splits[0][1] + '_nonempty.txt')
        not_empty_stats_file = os.path.join(self.root, 'ImageSets', 'VID', self._splits[0][1] + '_nonempty_stats.txt')

        if os.path.exists(not_empty_file):  # if the splits file exists
            logging.info("Loading splits from: {}".format(not_empty_file))
            with open(not_empty_file, 'r') as f:
                good_sample_ids = [int(line.rstrip()) for line in f.readlines()]

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

    def _load_samples(self):
        """
        Load the samples of this dataset using the settings supplied

        Returns:
            dict : a dict of either video (videos=True) or frame (videos=False) samples

        """
        ids = list()
        for year, split in self._splits:

            # load the splits file
            logging.info("Loading splits from: {}".format(os.path.join(self.root, 'ImageSets', 'VID', split + '.txt')))
            with open(os.path.join(self.root, 'ImageSets', 'VID', split + '.txt'), 'r') as f:
                ids_ = [(int(line.split()[1]), self.root, split, line.split()[0]) for line in f.readlines()]

            # use only the 2015 samples
            if year == 2015:
                ids_ = [id_ for id_ in ids_ if 'ILSVRC2015' in id_[-1]]

            ids += ids_

        # Build a dictionary of videos {vid_id: (path str, split str, vid_id str, frames list, frame_ids list), ...}
        videos = dict()
        past_vid_id = ''
        frames = None
        frame_ids = None
        past_id = None
        for i, id_ in enumerate(ids):
            vid_id = id_[3][:-7]
            frame = id_[3][-6:]

            if i == 0:  # fist iteration - set the variables
                past_id = id_
                frames = [frame]
                frame_ids = [id_[0]]
                past_vid_id = vid_id

            elif i == len(ids)-1:  # last iteration - add the last video
                if self._frames < 1:  # cut down per video
                    frames = [frames[i] for i in range(0, len(frames), int(1 / self._frames))]
                    frame_ids = [frame_ids[i] for i in range(0, len(frame_ids), int(1 / self._frames))]
                elif self._frames > 1:  # cut down per video
                    frames = [frames[i] for i in range(0, len(frames), int(math.ceil(len(frames) / self._frames)))]
                    frame_ids = [frame_ids[i] for i in
                                 range(0, len(frame_ids), int(math.ceil(len(frames) / self._frames)))]

                videos[vid_id] = (id_[1], id_[2], vid_id, frames, frame_ids)

            else:
                if vid_id != past_vid_id:  # new video - add it

                    if self._frames < 1:  # cut down per video
                        frames = [frames[i] for i in range(0, len(frames), int(1/self._frames))]
                        frame_ids = [frame_ids[i] for i in range(0, len(frame_ids), int(1/self._frames))]
                    elif self._frames > 1:  # cut down per video
                        frames = [frames[i] for i in range(0, len(frames), int(math.ceil(len(frames)/self._frames)))]
                        frame_ids = [frame_ids[i] for i in
                                     range(0, len(frame_ids), int(math.ceil(len(frames)/self._frames)))]

                    videos[past_vid_id] = (past_id[1], past_id[2], past_vid_id, frames, frame_ids)

                    past_id = id_
                    frames = [frame]
                    frame_ids = [id_[0]]
                    past_vid_id = vid_id

                else:  # same video - just append the frames
                    frames.append(frame)
                    frame_ids.append(id_[0])

        # if we want samples to be full videos, return the dictionary of videos
        if self._videos:
            return videos
        else:  # we want frames, or windows of frames

            # reconstruct the frame samples from vid video samples
            frames = dict()
            for video in videos.values():
                for frame_name, frame_id in zip(video[3], video[4]):
                    frames[frame_id] = (video[0], video[1], os.path.join(video[2], frame_name))

            # build a temporal window of frames around each sample, adheres to the 'self._frames param' only containing
            # frames that are in the cut down set... so step is actually step*the_dataset_step
            if self._window_size > 1:
                self._windows = dict()

                for video in videos.values():  # lock to only getting frames from the same video
                    frame_ids = video[4]  # get the list of frame ids

                    for i in range(len(frame_ids)):  # for each frame in this video
                        window = list()  # setup a new window (will hold the frame_ids)

                        window_size = int(self._window_size / 2.0)  # halves and floors it

                        # lets step back through the video and get the frame_ids
                        for back_i in range(window_size*self._window_step, self._window_step-1, -self._window_step):
                            window.append(frame_ids[max(0, i-back_i)])  # will add first frame to pad if window too big

                        # add the current frame
                        window.append(frame_ids[i])

                        # lets step forward and add future frame_ids
                        for forward_i in range(self._window_step, window_size*self._window_step+1, self._window_step):
                            if len(window) == self._window_size:
                                break  # only triggered if self._window_size is even, we disregard last frame
                            # will add last frame to pad if window too big
                            window.append(frame_ids[min(len(frame_ids)-1, i+forward_i)])

                        # add the window frame ids to a dict to be used in the get() function
                        self._windows[frame_ids[i]] = window

            return frames

    def _load_label(self, idx, frame_id=None):
        """
        Parse the xml annotation files for a sample

        Args:
            idx (int): the sample index
            frame_id (str): needed if videos=True, will get the label for this particular frame

        Returns:
            numpy.ndarray : labels of shape (n, 6) - [[xmin, ymin, xmax, ymax, cls_id, trk_id], ...]
        """

        sample_id = self.sample_ids[idx]
        sample = self.samples[sample_id]

        anno_path = self._annotations_path.format(*sample)
        if self._videos:
            assert frame_id is not None
            img_id = (sample[0], sample[1], os.path.join(sample[2], frame_id))
            anno_path = self._annotations_path.format(*img_id)

        if not os.path.exists(anno_path):
            return np.array([[-1, -1, -1, -1, -1, -1]])

        root = et.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)

        # store the shapes for later usage
        if sample_id not in self._im_shapes:
            self._im_shapes[sample_id] = (width, height)

        label = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.wn_classes:
                continue
            cls_id = self.index_map[cls_name]
            trk_id = int(obj.find('trackid').text)
            xml_box = obj.find('bndbox')

            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)

            # validate the label so it fits in the image and has a positive area
            xmin, ymin, xmax, ymax = self._validate_label(xmin, ymin, xmax, ymax, width, height, anno_path)
            label.append([xmin, ymin, xmax, ymax, cls_id, trk_id])

        # if we didn't find a box label we need to add one
        if self._allow_empty and len(label) < 1:
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

    @staticmethod
    def _pad_to_dense(labels, maxlen=100):
        """
        pad a labels numpy array to have a length maxlen
        Args:
            labels (numpy.ndarray): the array to pad
            maxlen: (int): the padding size (default is 100)

        Returns:
            numpy.ndarray: a padded array
        """

        x = -np.ones((maxlen, 6))
        for enu, row in enumerate(labels):
            x[enu, :] += row + 1
        return x

    # def image_size(self, sample_id):
    #     if len(self._im_shapes) == 0:
    #         for idx in tqdm(range(len(self.sample_ids)), desc="populating im_shapes"):
    #             self._load_label(idx)
    #     return self._im_shapes[sample_id]

    def stats(self):
        """
        Get the dataset statistics

        Returns:
            str: an output string with all of the information
            list: a list of counts of the number of boxes per class

        """
        cls_boxes = []
        n_samples = len(self.sample_ids)
        n_boxes = [0]*len(self.classes)
        n_instances = [0]*len(self.classes)
        past_vid_id = ''
        for idx in tqdm(range(len(self.sample_ids)), desc="Calculating stats"):
            sample_id = self.sample_ids[idx]
            vid_id = self.samples[sample_id][2]

            if vid_id != past_vid_id:
                vid_instances = []
                past_vid_id = vid_id
            if self._videos:
                for frame_id in self.samples[sample_id][3]:
                    for box in self._load_label(idx, frame_id):
                        n_boxes[int(box[4])] += 1
                        if int(box[5]) not in vid_instances:
                            vid_instances.append(int(box[5]))
                            n_instances[int(box[4])] += 1
            else:
                for box in self._load_label(idx):
                    if int(box[4]) < 0:  # not actually a box
                        continue
                    n_boxes[int(box[4])] += 1
                    if int(box[5]) not in vid_instances:
                        vid_instances.append(int(box[5]))
                        n_instances[int(box[4])] += 1

        if self._videos:
            out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n{8: <10} {9}\n'.format(
                'Split:', ', '.join([str(s[0]) + s[1] for s in self._splits]),
                'Videos:', n_samples,
                'Boxes:', sum(n_boxes),
                'Instances:', sum(n_instances),
                'Classes:', len(self.classes))
        else:
            out_str = '{0: <10} {1}\n{2: <10} {3}\n{4: <10} {5}\n{6: <10} {7}\n{8: <10} {9}\n'.format(
                'Split:', ', '.join([str(s[0]) + s[1] for s in self._splits]),
                'Frames:', n_samples,
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
            filename = self._image_path.format(*self.samples[sample_id])
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


def iou(bb, bbgt):
    """
    Single box IoU calculation function for generate_motion_ious

    Args:
        bb: bounding box 1
        bbgt: bounding box 2

    Returns:
        float: the IoU between the two boxes
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


def generate_motion_ious():
    """
    Used to generate motion_ious .json files matching that of imagenet_vid_groundtruth_motion_iou.mat from FGFA
    Except these are keyed on the image ids listed beside each frame in the ImageSets

    Note This script is relatively intensive
    
    Returns:
        None

    """
    for split in ['train', 'val']:
        dataset = ImageNetVidDetection(splits=[(2017, split)], allow_empty=True, videos=True, percent=1)
    
        all_ious = {}  # using dict for better removing of elements on loading based on sample_id
        sample_id = 1  # messy but works
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

                if frame_ious:  # if this frame has no boxes we add a 0.0
                    all_ious[sample_id] = frame_ious
                else:
                    all_ious[sample_id] = [0.0]
                sample_id += 1
    
        with open(os.path.join(dataset.root, '{}_motion_ious.json'.format(split)), 'w') as f:
            json.dump(all_ious, f)


if __name__ == '__main__':
    train_dataset = ImageNetVidDetection(splits=[(2017, 'train')], allow_empty=False, frames=0.04)
    for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
        pass
    print(train_dataset)

    val_dataset = ImageNetVidDetection(splits=[(2017, 'val')], allow_empty=False, frames=0.04, window_size=5)
    for s in tqdm(val_dataset, desc='Test Pass of Validation Set'):
        pass
    print(val_dataset)

    test_dataset = ImageNetVidDetection(splits=[(2015, 'test')], allow_empty=True)
    for s in tqdm(test_dataset, desc='Test Pass of Testing Set'):
        pass
    print(test_dataset)
