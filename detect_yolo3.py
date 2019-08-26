from __future__ import division
from __future__ import print_function

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import glob
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
import mxnet as mx
from mxnet import gluon
import logging
import numpy as np
import os
import random
from tqdm import tqdm

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection
from datasets.detectset import DetectSet

from metrics.pascalvoc import VOCMApMetric
from metrics.mscoco import COCODetectionMetric
from metrics.imgnetvid import VIDDetectionMetric

from models.transforms import YOLO3DefaultInferenceTransform

from utils.general import as_numpy
from utils.image import cv_plot_bbox
from utils.video import video_to_frames

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

logging.basicConfig(level=logging.INFO)

flags.DEFINE_string('model_path', 'yolo3_darknet53_voc_best.params',
                    'Path to the detection model to use')
flags.DEFINE_string('network', 'darknet53',
                    'Base network name: darknet53 or mobilenet1.0.')
flags.DEFINE_string('dataset', 'voc',
                    'Dataset or .jpg image or .mp4 video or .txt image/video list.')
flags.DEFINE_string('trained_on', '',
                    'Dataset the model was trained on.')
flags.DEFINE_string('save_prefix', '0001',
                    'Model save prefix.')
flags.DEFINE_string('save_dir', 'results',
                    'Save directory to save images.')
flags.DEFINE_list('metrics', ['voc', 'coco'],
                  'List of metrics separated by , eg. voc,coco')
flags.DEFINE_integer('batch_size', 1,
                     'Batch size for detection: higher faster, but more memory intensive.')
flags.DEFINE_integer('data_shape', 416,
                     'Input data shape.')
flags.DEFINE_float('detection_threshold', 0.5,
                   'The threshold on detections to them being displayed.')
flags.DEFINE_integer('max_do', -1,
                     'Maximum samples to detect on. -1 is all.')
flags.DEFINE_float('frames', 0.04,
                   'Based per video - and is NOT randomly sampled:'
                   'If <1: Percent of the full dataset to take eg. .04 (every 25th frame) - range(0, len(video), int(1/frames))'
                   'If >1: This many frames per video - range(0, len(video), int(ceil(len(video)/frames)))'
                   'If =1: Every sample used - full dataset')

flags.DEFINE_boolean('visualise', False,
                     'Do you want to display the detections?')
flags.DEFINE_boolean('display_gt', True,
                     'Do you want to display the ground truth boxes on the images?')

flags.DEFINE_list('gpus', [0],
                  'GPU IDs to use. Use comma or space for multiple eg. 0,1 or 0 1.')
flags.DEFINE_integer('num_workers', 8,
                     'The number of workers should be picked so that it’s equal to number of cores on your machine'
                     ' for max parallelization.')


def get_dataset(dataset_name):
    if dataset_name.lower() == 'voc':
        dataset = VOCDetection(splits=[(2007, 'test')], inference=True)

    elif dataset_name.lower() == 'coco':
        dataset = COCODetection(splits=['instances_val2017'], allow_empty=True, inference=True)

    elif dataset_name.lower() == 'det':
        dataset = ImageNetDetection(splits=['val'], allow_empty=False, inference=True)

    elif dataset_name.lower() == 'vid':
        dataset = ImageNetVidDetection(splits=[(2017, 'val')], allow_empty=True, frames=FLAGS.frames, inference=True)

    elif dataset_name[-4:] == '.txt':  # list of images or list of videos
        with open(dataset_name, 'r') as f:
            files = [l.rstrip() for l in f.readlines()]
        if files[0][-4:] == '.mp4':  # list of videos
            img_list = list()
            for file in files:  # make frames in tmp folder
                frames_dir = video_to_frames(file, os.path.join('data', 'tmp'),
                                             os.path.join('data', 'tmp', 'stats'), overwrite=False)

                img_list += glob.glob(frames_dir + '/**/*.jpg', recursive=True)

        elif files[0][-4:] == '.jpg':  # list of images
            img_list = files
        dataset = DetectSet(img_list)

    elif dataset_name[-4:] == '.jpg':  # single image
        dataset = DetectSet([dataset_name])

    elif dataset_name[-4:] == '.mp4':
        # make frames in tmp folder
        frames_dir = video_to_frames(dataset_name, os.path.join('data', 'tmp'),
                                     os.path.join('data', 'tmp', 'stats'), overwrite=False)
        img_list = glob.glob(frames_dir + '/**/*.jpg', recursive=True)
        dataset = DetectSet(img_list)

    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset_name))

    return dataset


def get_dataloader(dataset, batch_size):
    width, height = FLAGS.data_shape, FLAGS.data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1), Stack())
    loader = gluon.data.DataLoader(dataset.transform(YOLO3DefaultInferenceTransform(width, height, 3)),
                                   batch_size, False, last_batch='keep', num_workers=FLAGS.num_workers,
                                   batchify_fn=batchify_fn)
    return loader


def get_metric(dataset, metric_name, data_shape, save_dir, class_map=None):
    if metric_name.lower() == 'voc':
        metric = VOCMApMetric(iou_thresh=0.5, class_names=dataset.classes, class_map=class_map)

    elif metric_name.lower() == 'coco':
        metric = COCODetectionMetric(dataset, save_dir, cleanup=True, data_shape=None)

    elif metric_name.lower() == 'vid':
        metric = VIDDetectionMetric(dataset, iou_thresh=0.5, data_shape=None, class_map=class_map)

    else:
        raise NotImplementedError('Mertic: {} not implemented.'.format(metric_name))

    return metric


def detect(net, dataset, loader, ctx, max_do=-1):
    net.collect_params().reset_ctx(ctx)
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    boxes = dict()
    if max_do < 0:
        max_do = len(dataset)
    c = 0
    with tqdm(total=min(max_do, len(dataset)), desc="Detecting") as pbar:
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

            for id, score, box, sidx in zip(*[as_numpy(x) for x in [det_ids, det_scores, det_bboxes, sidxs]]):

                file = dataset.sample_path(int(sidx))

                valid_pred = np.where(id.flat >= 0)[0]  # get the boxes that have a class assigned
                box = box[valid_pred, :] / batch[0].shape[2]  # normalise boxes
                id = id.flat[valid_pred].astype(int)
                score = score.flat[valid_pred]

                for id_, box_, score_ in zip(id, box, score):
                    if file in boxes:
                        boxes[file].append([id_, score_]+list(box_))
                    else:
                        boxes[file] = [[id_, score_]+list(box_)]

            pbar.update(batch[0].shape[0])
            c += batch[0].shape[0]
            if c > max_do:
                break

    return boxes


def save_predictions(save_dir, dataset, boxes, overwrite=True, max_do=-1):
    if not overwrite and os.path.exists(os.path.join(save_dir, 'gt')) and os.path.exists(os.path.join(save_dir, 'pred')):
        logging.info("Ground truth and prediction files already exist")

    os.makedirs(os.path.join(save_dir, 'pred'), exist_ok=True)

    if max_do < 0:
        max_do = len(dataset)

    for idx in tqdm(range(min(len(dataset), max_do)), desc="Saving out prediction .txts"):
        img_path = dataset.sample_path(idx)

        file_id = img_path.split('/')[-1][:-4]
        if FLAGS.dataset == 'vid':
            file_id = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1][:-5])
            os.makedirs(os.path.join(save_dir, 'pred', img_path.split('/')[-2]), exist_ok=True)

        with open(os.path.join(save_dir, 'pred', file_id + '.txt'), 'w') as f:
            if img_path in boxes:
                for box in boxes[img_path]:  # sid, class, score, box
                    f.write("{},{},{},{},{},{},{}\n".format(img_path, box[0], box[1], box[2], box[3], box[4], box[5]))


def load_predictions(save_dir, dataset, max_do=-1):
    if not os.path.exists(os.path.join(save_dir, 'pred')):
        logging.error("Predictions directory does not exist {}".format(os.path.join(save_dir, 'pred')))
        return None

    boxes = dict()
    for idx in tqdm(range(min(len(dataset), max_do)), desc="Loading in prediction .txts"):
        img_path = dataset.sample_path(idx)

        file_id = img_path.split('/')[-1][:-4]
        if FLAGS.dataset == 'vid':
            file_id = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1][:-5])

        if not os.path.exists(os.path.join(save_dir, 'pred', file_id + '.txt')):
            logging.error("Prediction file does not exist {}".format(os.path.join(save_dir, 'pred', file_id + '.txt')))
            return None

        with open(os.path.join(save_dir, 'pred', file_id + '.txt'), 'r') as f:
            bb = [line.rstrip().split(',') for line in f.readlines()]
        for box in bb:
            if box[0] in boxes:
                boxes[box[0]].append([int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])])
            else:
                boxes[box[0]] = [[int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])]]

    return boxes


def visualise_predictions(save_dir, dataset, trained_on_dataset, boxes,
                          max_do=-1, display_gt=False, detection_thresh=0.5):
    colors = dict()
    for i in range(200):
        colors[i] = (int(256 * random.random()), int(256 * random.random()), int(256 * random.random()))
    colors_gt = dict()
    for i in range(200):
        colors_gt[i] = (0, 255, 0)

    if max_do < 0:
        max_do = len(dataset)

    for idx in tqdm(range(min(len(dataset), max_do)), desc="Saving out images"):

        img_path = dataset.sample_path(idx)
        img = cv2.imread(img_path)

        imgb, y, _ = dataset[idx]

        if display_gt and len(y) > 0:
            img = cv_plot_bbox(img=img,
                               bboxes=[list(g) for g in y[:, :4]],
                               scores=[1]*len(y),
                               labels=[g for g in y[:, 4]],
                               thresh=detection_thresh,
                               colors=colors_gt,
                               class_names=dataset.classes,
                               absolute_coordinates=True)

        if img_path in boxes:
            img = cv_plot_bbox(img=img,
                               bboxes=[b[2:] for b in boxes[img_path]],
                               scores=[b[1] for b in boxes[img_path]],
                               labels=[b[0] for b in boxes[img_path]],
                               thresh=detection_thresh,
                               colors=colors,
                               class_names=trained_on_dataset.classes,
                               absolute_coordinates=False)

        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
        if FLAGS.dataset == 'vid':
            os.makedirs(os.path.join(save_dir, 'vis', img_path.split('/')[-2]), exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, 'vis', '/'.join(img_path.split('/')[-2:])), img)
        else:
            cv2.imwrite(os.path.join(save_dir, 'vis', img_path.split('/')[-1]), img)


def evaluate(metrics, dataset, predictions):
    for idx in tqdm(range(len(dataset)), desc="Evaluating with metrics"):

        img_path = dataset.sample_path(idx)

        # get the gt boxes : [n_gpu, batch_size, samples, dim] : [1, 1, ?, 4 or 1]
        img, y, _ = dataset[idx]
        gt_bboxes = [np.expand_dims(y[:, :4], axis=0)]
        gt_ids = [np.expand_dims(y[:, 4], axis=0)]
        gt_difficults = [np.expand_dims(y[:, 5], axis=0) if y.shape[-1] > 5 else None]

        # get the predictions : [n_gpu, batch_size, samples, dim] : [1, 1, ?, 4 or 1]
        if img_path in predictions:
            det_bboxes = [[[[b[2]*img.shape[1],  # change pred box dims to match image (unnormalise them)
                             b[3]*img.shape[0],
                             b[4]*img.shape[1],
                             b[5]*img.shape[0]] for b in predictions[img_path]]]]
            det_ids = [[[[b[0]] for b in predictions[img_path]]]]
            det_scores = [[[[b[1]] for b in predictions[img_path]]]]

        for metric in metrics:
            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)

    return [metric.get() for metric in metrics]


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

    # if we aren't given a full path, assume the file is in 'models/save_prefix' directory
    if len(os.path.split(FLAGS.model_path)[0]) > 0:
        model_path = FLAGS.model_path
    else:
        model_path = os.path.join('models', FLAGS.save_prefix, FLAGS.model_path)

    # check model exists
    if not os.path.exists(model_path):
        logging.error("Model doesn't appear where it's expected: {}".format(model_path))

    # get dataset
    dataset = get_dataset(FLAGS.dataset)

    # for use when model predictions are different to evaluation set classes
    if FLAGS.trained_on:
        trained_on_dataset = get_dataset(FLAGS.trained_on)
    else:
        trained_on_dataset = dataset

    # fix for tiny datasets of 1 or few elements
    batch_size = FLAGS.batch_size
    if len(dataset) < batch_size:
        batch_size = len(dataset)

    # handle gpu usage
    gpus = FLAGS.gpus
    if batch_size < len(gpus):
        gpus = [gpus[0]]

    # contexts
    ctx = [mx.gpu(int(i)) for i in gpus]
    ctx = ctx if ctx else [mx.cpu()]

    # dataloader
    loader = get_dataloader(dataset, batch_size)

    # setup network
    net_name = '_'.join(('yolo3', FLAGS.network, 'custom'))
    net = get_model(net_name, root='models', pretrained_base=True, classes=trained_on_dataset.classes)
    net.load_parameters(model_path)

    max_do = FLAGS.max_do
    if max_do < 0:
        max_do = len(dataset)

    # organise the save directories for the results
    if FLAGS.dataset in ['voc', 'coco', 'det', 'vid']:
        save_dir = os.path.join('models', FLAGS.save_prefix, FLAGS.save_dir, FLAGS.dataset)
    else:
        save_dir = os.path.join('models', FLAGS.save_prefix, FLAGS.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # attempt to load predictions
    predictions = load_predictions(save_dir, dataset, max_do=max_do)

    if predictions is None:  # id not exist detect and make
        predictions = detect(net, dataset, loader, ctx, max_do=max_do)  # todo fix det thresh
        save_predictions(save_dir, dataset, predictions)

    if FLAGS.visualise:
        visualise_predictions(save_dir, dataset, trained_on_dataset, predictions,
                              max_do, display_gt=FLAGS.display_gt, detection_threshold=FLAGS.detection_threshold)

    metrics = list()
    if FLAGS.metrics:
        for metric_name in FLAGS.metrics:
            if FLAGS.trained_on:  # for use when model preds are diff to eval set classes
                metrics.append(get_metric(dataset, metric_name, FLAGS.data_shape, save_dir,
                                          class_map=get_class_map(trained_on_dataset, dataset)))
            else:
                metrics.append(get_metric(dataset, metric_name, FLAGS.data_shape, save_dir))

        results = evaluate(metrics, dataset, predictions)

        for m, metric_name in enumerate(FLAGS.metrics):
            names, values = results[m]
            with open(os.path.join(save_dir, metric_name+'.txt'), 'w') as f:
                for k, v in zip(names, values):
                    print(k, v)
                    f.write('{} {}\n'.format(k, v))


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass
