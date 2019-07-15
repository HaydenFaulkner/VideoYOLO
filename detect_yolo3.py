from __future__ import division
from __future__ import print_function

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import logging
import mxnet as mx
from mxnet import gluon
from gluoncv.model_zoo import get_model
import gluoncv as gcv
from gluoncv.data.batchify import Tuple, Stack, Pad
import numpy as np
import random
from tqdm import tqdm

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection
from datasets.detectset import DetectSet

from utils.general import as_numpy, YOLO3DefaultInferenceTransform
from utils.image import cv_plot_bbox
from utils.video import video_to_frames

logging.basicConfig(level=logging.INFO)


def get_dataset(dataset_name):  # todo add detection flag to the datasets so the get function behaves as desired
    if dataset_name.lower() == 'voc':
        dataset = VOCDetection(root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'), splits=[(2007, 'test')], inference=True)
    elif dataset_name.lower() == 'coco':
        dataset = COCODetection(root=os.path.join('datasets', 'MSCoco'),
                                splits='instances_val2017', skip_empty=False, inference=True)
    elif dataset_name.lower() == 'det':
        dataset = ImageNetDetection(root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
                                    splits=['val'], allow_empty=False, inference=True)
    elif dataset_name.lower() == 'vid':
        dataset = ImageNetVidDetection(root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
                                       splits=[(2017, 'val')], allow_empty=False, videos=False, frames=0.2, inference=True)
    elif dataset_name[-4:] == '.txt':  # list of images or list of videos
        with open(dataset_name, 'r') as f:
            files = [l.rstrip() for l in f.readlines()]
        if files[0][-4:] == '.mp4':  # list of videos
            img_list = list()
            for file in files:  # make frames in tmp folder
                img_list += video_to_frames(file, os.path.join('data', 'tmp'),
                                            os.path.join('data', 'tmp', 'stats'), overwrite=False)
        elif files[0][-4:] == '.jpg':  # list of images
            img_list = files
        dataset = DetectSet(img_list)
    elif dataset_name[-4:] == '.jpg':  # single image
        dataset = DetectSet([dataset_name])
    elif dataset_name[-4:] == '.mp4':
        # make frames in tmp folder
        img_list = video_to_frames(dataset_name, os.path.join('data', 'tmp'),
                                   os.path.join('data', 'tmp', 'stats'), overwrite=False)
        dataset = DetectSet(img_list)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset_name))
    return dataset


def get_dataloader(dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1), Stack())  # todo ensure this is correct
    loader = gluon.data.DataLoader(dataset.transform(YOLO3DefaultInferenceTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers, batchify_fn=batchify_fn)
    return loader


def detect(net, dataset, loader, ctx, detection_threshold=0, max_do=-1):
    net.collect_params().reset_ctx(ctx)
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    boxes = dict()
    gt_boxes = dict()
    if max_do < 0:
        max_do = len(dataset)
    c = 0
    with tqdm(total=min(max_do, len(dataset))) as pbar:
        for ib, batch in enumerate(loader):

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            idxs = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            sidxs = []
            for x, y, sidx in zip(data, label, idxs):
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
                sidxs.append(sidx)

            for id, score, box, gid, gbox, sidx in zip(*[as_numpy(x) for x in [det_ids, det_scores, det_bboxes,
                                                                               gt_ids, gt_bboxes, sidxs]]):

                file = dataset.sample_path(int(sidx))

                valid_pred = np.where(id.flat >= 0)[0]  # get the boxes that have a class assigned
                box = box[valid_pred, :] / batch[0].shape[2]  # normalise boxes
                id = id.flat[valid_pred].astype(int)
                score = score.flat[valid_pred]

                for id_, box_, score_ in zip(id, box, score):
                    if score_ > detection_threshold:
                        if file in boxes:
                            boxes[file].append([id_, score_]+list(box_))
                        else:
                            boxes[file] = [[id_, score_]+list(box_)]

                valid_gt = np.where(gid.flat >= 0)[0]
                gbox = gbox[valid_gt, :] / batch[0].shape[2]
                gid = gid.flat[valid_gt].astype(int)

                for gid_, gbox_ in zip(gid, gbox):
                    if file in gt_boxes:
                        gt_boxes[file].append([gid_]+list(gbox_))
                    else:
                        gt_boxes[file] = [[gid_]+list(gbox_)]

            pbar.update(batch[0].shape[0])
            c += batch[0].shape[0]
            if c > max_do:
                break

    return boxes, gt_boxes


def get_class_map(trained_on, eval_on):
    toc = trained_on.wn_classes
    eoc = eval_on.wn_classes

    class_map = []
    for c in eoc:
        if c in toc:
            class_map.append(toc.index(c))
        else:
            class_map.append(-1)

    return class_map


def main(_argv):

    # check model exists
    model_path = os.path.join('models', FLAGS.save_prefix, FLAGS.model_path)
    if not os.path.exists(model_path):
        logging.error("Model doesn't appear where it's expected: {}".format(model_path))

    # dataset
    dataset = get_dataset(FLAGS.dataset)

    if FLAGS.trained_on: # for use when model preds are diff to eval set classes
        trained_on_dataset = get_dataset(FLAGS.trained_on)
    else:
        trained_on_dataset = dataset

    # fix for tiny datasets of 1 or few elements
    batch_size = FLAGS.batch_size
    if len(dataset) < batch_size:
        batch_size = len(dataset)

    gpus = FLAGS.gpus.split(',')
    if batch_size < len(gpus):
        gpus = [gpus[0]]

    # contexts
    ctx = [mx.gpu(int(i)) for i in gpus if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # dataloader
    loader = get_dataloader(dataset, FLAGS.data_shape, batch_size, FLAGS.num_workers)

    # network
    net_name = '_'.join(('yolo3', FLAGS.network, 'custom'))
    net = get_model(net_name, root='models', pretrained_base=True, classes=trained_on_dataset.classes)
    net.load_parameters(model_path)

    max_do = FLAGS.max_do
    if max_do < 0:
        max_do = len(dataset)

    # detect
    boxes, gt_boxes = detect(net, dataset, loader, ctx, detection_threshold=FLAGS.detection_threshold, max_do=max_do)

    colors = dict()
    for i in range(200):
        colors[i] = (int(256 * random.random()), int(256 * random.random()), int(256 * random.random()))
    colors_gt = dict()
    for i in range(200):
        colors_gt[i] = (0, 255, 0)

    if FLAGS.dataset in ['voc', 'coco', 'det', 'vid']:
        save_path = os.path.join('models', FLAGS.save_prefix, FLAGS.save_dir, FLAGS.dataset)
    else:
        save_path = os.path.join('models', FLAGS.save_prefix, FLAGS.save_dir)
    os.makedirs(save_path, exist_ok=True)

    for idx in tqdm(range(min(len(dataset), max_do)), desc="Saving out images"):

        img_path = dataset.sample_path(idx)
        img = cv2.imread(img_path)

        if FLAGS.display_gt and img_path in gt_boxes:
            img = cv_plot_bbox(img=img,
                               bboxes=[gb[1:] for gb in gt_boxes[img_path]],
                               scores=[1 for gb in gt_boxes[img_path]],
                               labels=[gb[0] for gb in gt_boxes[img_path]],
                               thresh=0,
                               colors=colors_gt,
                               class_names=dataset.classes,
                               absolute_coordinates=False)

        if img_path in boxes:
            img = cv_plot_bbox(img=img,
                               bboxes=[b[2:] for b in boxes[img_path]],
                               scores=[b[1] for b in boxes[img_path]],
                               labels=[b[0] for b in boxes[img_path]],
                               thresh=0,
                               colors=colors,
                               class_names=trained_on_dataset.classes,
                               absolute_coordinates=False)

        if FLAGS.dataset == 'vid':
            os.makedirs(os.path.join(save_path, img_path.split('/')[-2]), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, '/'.join(img_path.split('/')[-2:])), img)
        else:
            cv2.imwrite(os.path.join(save_path, img_path.split('/')[-1]), img)


if __name__ == '__main__':

    flags.DEFINE_string('model_path', 'yolo3_darknet53_voc_best.params',
                        'Path to the detection model to use')
    flags.DEFINE_string('network', 'darknet53',
                        'Base network name: darknet53 or mobilenet1.0.')
    flags.DEFINE_string('dataset', 'voc',
                        'Dataset or .jpg image or .mp4 video or .txt image/video list.')
    flags.DEFINE_string('trained_on', 'voc',
                        'Dataset the model was trained on.')
    flags.DEFINE_string('save_prefix', '0001',
                        'Model save prefix.')
    flags.DEFINE_string('save_dir', 'vis',
                        'Save directory to save images.')
    flags.DEFINE_integer('batch_size', 2,
                         'Batch size for detection: higher faster, but more memory intensive.')
    flags.DEFINE_integer('data_shape', 416,
                         'Input data shape.')
    flags.DEFINE_float('detection_threshold', 0.5,
                       'The threshold on detections to them being displayed.')
    flags.DEFINE_integer('max_do', 5000,
                         'Maximum samples to detect on. -1 is all.')

    flags.DEFINE_boolean('display_gt', True,
                         'Do you want to display the ground truth boxes on the images?')

    flags.DEFINE_string('gpus', '0',
                        'GPU IDs to use. Use comma for multiple eg. 0,1.')
    flags.DEFINE_integer('num_workers', 8,
                         'The number of workers should be picked so that it’s equal to number of cores on your machine'
                         ' for max parallelization.')

    try:
        app.run(main)
    except SystemExit:
        pass
