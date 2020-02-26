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

from models.definitions.yolo.transforms import YOLO3VideoInferenceTransform
from models.definitions.yolo.wrappers import yolo3_darknet53, yolo3_3ddarknet

from utils.general import as_numpy
from utils.image import cv_plot_bbox
from utils.video import video_to_frames

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

logging.basicConfig(level=logging.INFO)

flags.DEFINE_string('model_path', 'yolo3_darknet53_voc_best.params',
                    'Path to the detection model to use')
flags.DEFINE_string('network', 'darknet53',
                    'Base network name: darknet53')
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
flags.DEFINE_float('every', 25,
                   'do every this many frames')
flags.DEFINE_list('window', '1, 1',
                  'Temporal window size of frames and the frame gap/stride of the windows samples')
flags.DEFINE_string('k_join_type', None,
                    'way to fuse k type, either max, mean, cat.')
flags.DEFINE_string('k_join_pos', None,
                    'position of k fuse, either early or late.')
flags.DEFINE_string('block_conv_type', '2',
                    "convolution type for the YOLO blocks: '2'2D, '3':3D or '21':2+1D, must be used with 'late' joining")
flags.DEFINE_string('rnn_pos', None,
                    "position of RNN, currently only supports 'late' or 'out")
flags.DEFINE_string('corr_pos', None,
                    "position of correlation features calculation, currently only supports 'early' or 'late")
flags.DEFINE_integer('corr_d', 4,
                     'The d value for the correlation filter.')
flags.DEFINE_string('motion_stream', None,
                    'Add a motion stream? can be flownet or r21d.')
flags.DEFINE_string('stream_gating', None,
                    'Use gating on the appearence stream using the motion stream. can be add or mul.')
flags.DEFINE_list('conv_types', [2, 2, 2, 2, 2, 2],
                  'Darknet Conv types for layers, either 2, 21, or 3 D')
flags.DEFINE_string('h_join_type', None,
                    'Type to join hierarchical darknet. can be max or conv.')
flags.DEFINE_list('hier', [1, 1, 1, 1, 1],
                  'the hierarchical factors, the input must be temporally equal to all these multiplied together')
flags.DEFINE_boolean('mult_out', False,
                     'Have one or multiple outs for timeseries data')
flags.DEFINE_boolean('temp', False,
                     'Use new temporal model')

flags.DEFINE_boolean('visualise', False,
                     'Do you want to display the detections?')
flags.DEFINE_boolean('per_frame_metric', False,
                     'Do you want to save out a per frame metric to the prediction files?')
flags.DEFINE_string('worst_video_path', None,
                    'Path to save video of worst case detections. If not None will require visualise and '
                    'per_frame_metric to have been done previously')
flags.DEFINE_boolean('display_gt', True,
                     'Do you want to display the ground truth boxes on the images?')
flags.DEFINE_boolean('model_agnostic', False,
                     'make the model class agnostic?')
flags.DEFINE_boolean('metric_agnostic', False,
                     'make the metric class agnostic?')

flags.DEFINE_list('gpus', [0],
                  'GPU IDs to use. Use comma or space for multiple eg. 0,1 or 0 1.')
flags.DEFINE_integer('num_workers', 8,
                     'The number of workers should be picked so that it’s equal to number of cores on your machine'
                     ' for max parallelization.')
flags.DEFINE_boolean('new_model', False,
                     'Use features Yolo (new) or stages Yolo (old)?')
flags.DEFINE_integer('offset', 0,
                     'If mult_out specified this selects the offset to test. Can be -2, -1, 0, 1, 2')


def get_dataset(dataset_name):
    if dataset_name.lower() == 'voc':
        dataset = VOCDetection(splits=[(2007, 'test')], inference=True)

    elif dataset_name.lower() == 'coco':
        dataset = COCODetection(splits=['instances_val2017'], allow_empty=True, inference=True)

    elif dataset_name.lower() == 'det':
        dataset = ImageNetDetection(splits=['val'], allow_empty=False, inference=True)

    elif dataset_name.lower() == 'vid':
        dataset = ImageNetVidDetection(splits=[(2017, 'val')], allow_empty=True, every=FLAGS.every,
                                       window=FLAGS.window, inference=True, mult_out=FLAGS.mult_out)

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
    loader = gluon.data.DataLoader(dataset.transform(YOLO3VideoInferenceTransform(width, height)),
                                   batch_size, False, last_batch='keep', num_workers=FLAGS.num_workers,
                                   batchify_fn=batchify_fn)
    return loader


def get_metric(dataset, metric_name, data_shape, save_dir, class_map=None):
    if metric_name.lower() == 'voc':
        metric = VOCMApMetric(iou_thresh=0.5, class_names=dataset.classes, class_map=class_map)

    elif metric_name.lower() == 'coco':
        metric = COCODetectionMetric(dataset, save_dir, cleanup=True, data_shape=None)

    elif metric_name.lower() == 'vid':
        metric = VIDDetectionMetric(dataset, iou_thresh=0.5, data_shape=None, class_map=class_map,
                                    agnostic=FLAGS.metric_agnostic, offset=FLAGS.offset)

    else:
        raise NotImplementedError('Mertic: {} not implemented.'.format(metric_name))

    return metric


def detect(net, dataset, loader, ctx, max_do=-1):
    net.collect_params().reset_ctx(ctx)
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    # net.hybridize()
    boxes = dict()
    if FLAGS.mult_out:
        boxes = [dict(), dict(), dict(), dict(), dict()]
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
            # gt_bboxes = []
            # gt_ids = []
            # gt_difficults = []
            sidxs = []
            for x, y, sidx in zip(data, label, idxs):
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[-1]))
                # split ground truths
                # gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                # gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                # gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
                sidxs.append(sidx)

            for id, score, box, sidx in zip(*[as_numpy(x) for x in [det_ids, det_scores, det_bboxes, sidxs]]):

                if FLAGS.mult_out:
                    files = dataset.window_paths(int(sidx))

                    for offset, file in enumerate(files):
                        if offset != 2 and file == files[2]:
                            continue  # we skip the offset frames if they are the same as the central frame, prevents repeating the boundary frames

                        valid_pred = np.where(id[offset].flat >= 0)[0]  # get the boxes that have a class assigned
                        box_o = box[offset, valid_pred, :] / batch[0].shape[-1]  # normalise boxes
                        id_o = id[offset].flat[valid_pred].astype(int)
                        score_o = score[offset].flat[valid_pred]

                        for id_, box_, score_ in zip(id_o, box_o, score_o):
                            if file in boxes[offset]:
                                boxes[offset][file].append([id_, score_] + list(box_))
                            else:
                                boxes[offset][file] = [[id_, score_] + list(box_)]

                else:
                    file = dataset.sample_path(int(sidx))

                    valid_pred = np.where(id.flat >= 0)[0]  # get the boxes that have a class assigned
                    box = box[valid_pred, :] / batch[0].shape[-1]  # normalise boxes
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


def save_predictions(save_dir, dataset, boxes, overwrite=True, max_do=-1, agnostic=False):
    if agnostic:
        save_dir = os.path.join(save_dir, 'pred_ag')
    else:
        save_dir = os.path.join(save_dir, 'pred')

    if not overwrite and os.path.exists(save_dir):
        logging.info("Ground truth and prediction files already exist")

    os.makedirs(save_dir, exist_ok=True)

    if max_do < 0:
        max_do = len(dataset)

    for idx in tqdm(range(min(len(dataset), max_do)), desc="Saving out prediction .txts"):

        if FLAGS.mult_out:
            img_paths = dataset.window_paths(idx)

            for offset, img_path in enumerate(img_paths):

                file_id = img_path.split('/')[-1][:-4]
                if FLAGS.dataset == 'vid':
                    file_id = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1][:-5])
                    os.makedirs(os.path.join(save_dir, img_path.split('/')[-2]), exist_ok=True)

                with open(os.path.join(save_dir, file_id + '_' + str(offset-2) + '.txt'), 'w') as f:
                    if img_path in boxes[offset]:
                        for box in boxes[offset][img_path]:  # sid, class, score, box
                            f.write(
                                "{},{},{},{},{},{},{}\n".format(img_path, box[0], box[1], box[2], box[3], box[4], box[5]))
        else:
            img_path = dataset.sample_path(idx)

            file_id = img_path.split('/')[-1][:-4]
            if FLAGS.dataset == 'vid':
                file_id = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1][:-5])
                os.makedirs(os.path.join(save_dir, img_path.split('/')[-2]), exist_ok=True)

            with open(os.path.join(save_dir, file_id + '.txt'), 'w') as f:
                if img_path in boxes:
                    for box in boxes[img_path]:  # sid, class, score, box
                        f.write("{},{},{},{},{},{},{}\n".format(img_path, box[0], box[1], box[2], box[3], box[4], box[5]))


def load_predictions(save_dir, dataset, max_do=-1, metric=None, agnostic=False):
    if agnostic:
        save_dir = os.path.join(save_dir, 'pred_ag')
    else:
        save_dir = os.path.join(save_dir, 'pred')

    if metric is None:
        if not os.path.exists(save_dir):
            logging.error("Predictions directory does not exist {}".format(save_dir))
            return None
    else:
        if not os.path.exists(os.path.join(save_dir, 'metric')):
            logging.error("Predictions directory does not exist {}".format(os.path.join(save_dir, 'metric')))
            return None

    if FLAGS.mult_out:
        boxes = [dict(), dict(), dict(), dict(), dict()]
        for idx in tqdm(range(min(len(dataset), max_do)), desc="Loading in prediction .txts"):
            img_paths = dataset.window_paths(idx)

            for offset, img_path in enumerate(img_paths):

                file_id = img_path.split('/')[-1][:-4]
                if FLAGS.dataset == 'vid':
                    file_id = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1][:-5])

                add_metrics = False
                if metric is None:
                    file_path = os.path.join(save_dir, file_id + '_' + str(offset-2) + '.txt')
                    if not os.path.exists(file_path):
                        logging.error("Prediction file does not exist {}".format(file_path))
                        return None
                else:  # todo allow specific metrics
                    file_path = os.path.join(save_dir, 'metric', file_id + '_' + str(offset-2) + '.txt')
                    if not os.path.exists(file_path):
                        if not os.path.exists(os.path.join(save_dir, 'pred', file_id + '.txt')):
                            logging.error("Prediction file does not exist {}".format(file_path))
                            return None
                        else:
                            add_metrics = True

                with open(file_path, 'r') as f:
                    bb = [line.rstrip().split(',') for line in f.readlines()]

                if metric is not None and not add_metrics:
                    for box in bb:
                        if box[0] in boxes[offset]:
                            boxes[offset][box[0]].append([int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6]), float(box[7])])
                        else:
                            boxes[offset][box[0]] = [[int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6]), float(box[7])]]
                else:
                    for box in bb:
                        if box[0] in boxes[offset]:
                            boxes[offset][box[0]].append([int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])])
                        else:
                            boxes[offset][box[0]] = [[int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])]]

    else:
        boxes = dict()

        for idx in tqdm(range(min(len(dataset), max_do)), desc="Loading in prediction .txts"):
            img_path = dataset.sample_path(idx)

            file_id = img_path.split('/')[-1][:-4]
            if FLAGS.dataset == 'vid':
                file_id = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1][:-5])

            add_metrics = False
            if metric is None:
                file_path = os.path.join(save_dir, file_id + '.txt')
                if not os.path.exists(file_path):
                    logging.error("Prediction file does not exist {}".format(file_path))
                    return None
            else:  # todo allow specific metrics
                file_path = os.path.join(save_dir, 'metric', file_id + '.txt')
                if not os.path.exists(file_path):
                    if not os.path.exists(os.path.join(save_dir, 'pred', file_id + '.txt')):
                        logging.error("Prediction file does not exist {}".format(file_path))
                        return None
                    else:
                        add_metrics = True

            with open(file_path, 'r') as f:
                bb = [line.rstrip().split(',') for line in f.readlines()]

            if metric is not None and not add_metrics:
                for box in bb:
                    if box[0] in boxes:
                        boxes[box[0]].append([int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6]), float(box[7])])
                    else:
                        boxes[box[0]] = [[int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6]), float(box[7])]]
            else:
                for box in bb:
                    if box[0] in boxes:
                        boxes[box[0]].append([int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])])
                    else:
                        boxes[box[0]] = [[int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])]]

    if add_metrics:
        boxes = add_metrics_to_predictions(save_dir, dataset, metric)

    return boxes


def add_metrics_to_predictions(load_dir, dataset, metric):

    if not os.path.exists(load_dir):
        logging.error("Predictions directory does not exist {}".format(load_dir))
        return None

    summary = dict()
    boxes = dict()
    for idx in tqdm(range(len(dataset)), desc="Adding metrics to predictions .txt"):
        img_path = dataset.sample_path(idx)

        file_id = img_path.split('/')[-1][:-4]
        if FLAGS.dataset == 'vid':
            file_id = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1][:-5])

        if not os.path.exists(os.path.join(load_dir, file_id + '.txt')):
            logging.error("Prediction file does not exist {}".format(os.path.join(load_dir, file_id + '.txt')))
            return None

        # Load the predictions
        with open(os.path.join(load_dir, file_id + '.txt'), 'r') as f:
            bb = [line.rstrip().split(',') for line in f.readlines()]
        for box in bb:
            if box[0] in boxes:
                boxes[box[0]].append([int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])])
            else:
                boxes[box[0]] = [[int(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])]]

        # Run the metrics
        # get the gt boxes : [n_gpu, batch_size, samples, dim] : [1, 1, ?, 4 or 1]
        img, y, _ = dataset[idx]
        gt_bboxes = [np.expand_dims(y[:, :4], axis=0)]
        gt_ids = [np.expand_dims(y[:, 4], axis=0)]
        gt_difficults = [np.expand_dims(y[:, 5], axis=0) if y.shape[-1] > 5 else None]

        # get the predictions : [n_gpu, batch_size, samples, dim] : [1, 1, ?, 4 or 1]
        if img_path in boxes:
            det_bboxes = [[[[b[2] * img.shape[-2],  # change pred box dims to match image (unnormalise them)
                             b[3] * img.shape[-3],
                             b[4] * img.shape[-2],
                             b[5] * img.shape[-3]] for b in boxes[img_path]]]]
            det_ids = [[[[b[0]] for b in boxes[img_path]]]]
            det_scores = [[[[b[1]] for b in boxes[img_path]]]]

        metric.reset()
        metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        _, scores = metric.get()
        score = scores[-1]  # take the mean - which is the last score
        if FLAGS.dataset == 'vid':
            vid_id = img_path.split('/')[-2]
            if vid_id in summary:
                summary[vid_id].append(score)
            else:
                summary[vid_id] = [score]
        else:
            summary[img_path] = score

        # Save out the new detection file
        os.makedirs(os.path.join(load_dir, 'metric'), exist_ok=True)

        if FLAGS.dataset == 'vid':
            os.makedirs(os.path.join(load_dir, 'metric', img_path.split('/')[-2]), exist_ok=True)

        with open(os.path.join(load_dir, 'metric', file_id + '.txt'), 'w') as f:
            if img_path in boxes:
                for box in boxes[img_path]:  # sid, class, score, box
                    f.write("{},{},{},{},{},{},{},{}\n".format(img_path, box[0], box[1], box[2], box[3], box[4], box[5], score))
                    box.append(score)

    # generate a summary file listing ranking the worst clips
    if isinstance(summary[list(summary.keys())[0]], list):
        # need to sort on map first then number of frames, more frames ranked higher -> more wrong
        for k in summary.keys():
            summary[k] = [sum(summary[k])/len(summary[k]), len(summary[k])]
        summary_sorted = sorted(summary.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
        for i in range(len(summary_sorted)):
            summary_sorted[i] = (summary_sorted[i][0], summary_sorted[i][1][0])
    else:
        summary_sorted = sorted(summary.items(), key=lambda kv: kv[1])

    with open(os.path.join(load_dir, 'metric', 'summary.txt'), 'w') as f:
        for ss in summary_sorted:
            f.write("{}\t{}\n".format(ss[0], ss[1]))
    return boxes


def visualise_predictions(save_dir, dataset, trained_on_dataset, boxes,
                          max_do=-1, display_gt=False, detection_threshold=0.5):
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
                               scores=None,#[1]*len(y),
                               labels=[g for g in y[:, 4]],
                               thresh=detection_threshold,
                               colors=colors_gt,
                               class_names=dataset.classes,
                               absolute_coordinates=True)

        if img_path in boxes:
            img = cv_plot_bbox(img=img,
                               bboxes=[b[2:] for b in boxes[img_path]],
                               scores=[b[1] for b in boxes[img_path]],
                               labels=[b[0] for b in boxes[img_path]],
                               thresh=detection_threshold,
                               colors=colors,
                               class_names=trained_on_dataset.classes,
                               absolute_coordinates=False)

        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
        if FLAGS.dataset == 'vid':
            os.makedirs(os.path.join(save_dir, 'vis', img_path.split('/')[-2]), exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, 'vis', '/'.join(img_path.split('/')[-2:])), img)
        else:
            cv2.imwrite(os.path.join(save_dir, 'vis', img_path.split('/')[-1]), img)


def video_of_worst(video_path, frames_dir, summary_file=None, fps=4):
    assert fps < 25
    # add the .mp4 extension if it isn't already there
    if video_path[-4:] != ".mp4":
        video_path += ".mp4"

    files = list()
    summaries_dict = dict()
    # get the frame file paths
    if summary_file is None:
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            files = glob.glob(frames_dir + "/**/*" + ext, recursive=True)
            if len(files) > 0:
                break
    else:
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        summaries = [l.rstrip().split() for l in lines]

        for vid, score in summaries:
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                frame_files = glob.glob(os.path.join(frames_dir, vid) + "/**/*" + ext, recursive=True)
                if len(frame_files) > 0:
                    break
            # sort the files alphabetically assuming this will do them in the correct order
            frame_files.sort()
            files += frame_files
            summaries_dict[vid] = score

    # couldn't find any images
    if not len(files) > 0:
        print("Couldn't find any files in {}".format(frames_dir))
        return None

    # make specific frame size and fit all videos in this frame, with rescale and centering
    height = 1080
    width = 1920

    # create the videowriter - will create an .mp4
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))

    # load and write the frames to the video
    for filename in tqdm(files, desc="Generating Video {}".format(video_path)):
        blank_image = np.zeros((height, width, 3), np.uint8)
        if os.path.exists(filename):
            image = cv2.imread(filename)  # load the frame
        else:
            continue

        h, w, _ = image.shape
        ratio = min(height/h, width/w)
        hs = int(h*ratio)
        ws = int(w*ratio)
        hm = int(hs/2)
        wm = int(ws/2)
        height_m = int(height/2)
        width_m = int(width/2)

        image = cv2.resize(image, (ws, hs), interpolation=cv2.INTER_AREA)  # resize
        blank_image[height_m-hm:height_m+hm, width_m-wm:width_m+wm, :] = image[:2*hm, :2*wm, :]  # place in centre

        vid_id = filename.split('/')[-2]
        score = 'Clip AP: {:.2f}'.format(float(summaries_dict[vid_id]))

        cv2.putText(blank_image, '{:s}'.format(score), (1650, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(blank_image, '{:s}'.format(filename), (10, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        for _ in range(int(25/fps)):
            video.write(blank_image)  # write the frame to the video

    video.release()  # release the video

    return video_path


def evaluate(metrics, dataset, predictions):
    for idx in tqdm(range(len(dataset)), desc="Evaluating with metrics"):

        img_path = dataset.sample_path(idx)
        if FLAGS.mult_out:
            img_path = img_path[FLAGS.offset + 2]

        if img_path in predictions:
            # get the gt boxes : [n_gpu, batch_size, samples, dim] : [1, 1, ?, 4 or 1]
            img, y, _ = dataset[idx]
            if FLAGS.mult_out:
                img = img[FLAGS.offset+2]
                y = y[FLAGS.offset+2]
            gt_bboxes = [np.expand_dims(y[:, :4], axis=0)]
            gt_ids = [np.expand_dims(y[:, 4], axis=0)]
            gt_difficults = [np.expand_dims(y[:, 5], axis=0) if y.shape[-1] > 5 else None]

            # get the predictions : [n_gpu, batch_size, samples, dim] : [1, 1, ?, 4 or 1]
            det_bboxes = [[[[b[2]*img.shape[-2],  # change pred box dims to match image (unnormalise them)
                             b[3]*img.shape[-3],
                             b[4]*img.shape[-2],
                             b[5]*img.shape[-3]] for b in predictions[img_path]]]]
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

    FLAGS.window = [int(s) for s in FLAGS.window]
    FLAGS.conv_types = [int(s) for s in FLAGS.conv_types]
    FLAGS.hier = [int(s) for s in FLAGS.hier]
    if FLAGS.model_agnostic:
        FLAGS.metric_agnostic = True

    if FLAGS.motion_stream == 'flownet':
        FLAGS.data_shape = 384  # cause 416 is a nasty shape

    if FLAGS.window[0] > 1:
        assert FLAGS.dataset == 'vid', 'If using window size >1 you can only use the vid dataset'

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
        gpus = [int(gpus[0])]

    # contexts
    ctx = [mx.gpu(int(i)) for i in gpus]
    ctx = ctx if ctx else [mx.cpu()]

    # dataloader
    loader = get_dataloader(dataset, batch_size)

    # setup network
    # net_name = '_'.join(('yolo3', FLAGS.network, 'custom'))
    # net = get_model(net_name, root='models', pretrained_base=True, classes=trained_on_dataset.classes)
    if FLAGS.network == 'darknet53':
        if FLAGS.conv_types[0] is 2:
            net = yolo3_darknet53(trained_on_dataset.classes,
                                  k=FLAGS.window[0], k_join_type=FLAGS.k_join_type, k_join_pos=FLAGS.k_join_pos,
                                  block_conv_type=FLAGS.block_conv_type, rnn_pos=FLAGS.rnn_pos,
                                  corr_pos=FLAGS.corr_pos, corr_d=FLAGS.corr_d, motion_stream=FLAGS.motion_stream,
                                  agnostic=FLAGS.model_agnostic, add_type=FLAGS.stream_gating, new_model=FLAGS.new_model,
                                  hierarchical=FLAGS.hier, h_join_type=FLAGS.h_join_type, temporal=FLAGS.temp, t_out=FLAGS.mult_out)
        else:
            net = yolo3_3ddarknet(trained_on_dataset.classes, conv_types=FLAGS.conv_types)
    else:
        raise NotImplementedError('Backbone CNN model {} not implemented.'.format(FLAGS.network))
    net.initialize()
    if FLAGS.window[0] > 1:
        net.summary(mx.nd.random_normal(shape=(1, FLAGS.window[0], 3, FLAGS.data_shape, FLAGS.data_shape)))
    else:
        net.summary(mx.nd.random_normal(shape=(1, 3, FLAGS.data_shape, FLAGS.data_shape)))
    net.load_parameters(model_path)

    max_do = FLAGS.max_do
    if max_do < 0:
        max_do = len(dataset)

    # organise the save directories for the results
    if FLAGS.dataset in ['voc', 'coco', 'det', 'vid']:
        save_dir = os.path.join('models', 'experiments', FLAGS.save_prefix, FLAGS.save_dir, FLAGS.dataset)
    else:
        save_dir = os.path.join('models', 'experiments', FLAGS.save_prefix, FLAGS.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # attempt to load predictions

    per_sample_metric = None
    if FLAGS.worst_video_path is not None:
        FLAGS.per_frame_metric = True
    if FLAGS.per_frame_metric:
        per_sample_metric = get_metric(dataset, 'voc', FLAGS.data_shape, save_dir,
                                       class_map=get_class_map(trained_on_dataset, dataset))
    predictions = load_predictions(save_dir, dataset, max_do=max_do, metric=per_sample_metric,
                                   agnostic=FLAGS.model_agnostic)

    if predictions is None:  # id not exist detect and make
        predictions = detect(net, dataset, loader, ctx, max_do=max_do)  # todo fix det thresh
        save_predictions(save_dir, dataset, predictions, agnostic=FLAGS.model_agnostic)

    if FLAGS.mult_out:
        predictions = predictions[FLAGS.offset+2]

    if FLAGS.visualise:
        visualise_predictions(save_dir, dataset, trained_on_dataset, predictions,
                              max_do, display_gt=FLAGS.display_gt, detection_threshold=FLAGS.detection_threshold)

    if FLAGS.worst_video_path is not None:
        video_of_worst(FLAGS.worst_video_path, os.path.join(save_dir, "vis"),
                       summary_file=os.path.join(save_dir, 'metric', 'summary.txt'), fps=4)

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
            if FLAGS.metric_agnostic:
                metric_name += '_ag'
                if not FLAGS.model_agnostic:
                    metric_name += '_met'
            with open(os.path.join(save_dir, metric_name+'.txt'), 'w') as f:
                for k, v in zip(names, values):
                    print(k, v)
                    f.write('{} {}\n'.format(k, v))


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass
