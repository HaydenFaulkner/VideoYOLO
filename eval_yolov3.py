from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
from mxnet.gluon.nn import BatchNorm
import gluoncv as gcv
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.model_zoo.yolo.yolo3 import get_yolov3
from gluoncv.model_zoo.yolo.darknet import darknet53
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
# from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from metrics.mscoco import COCODetectionMetric
from metrics.imgnetvid import VIDDetectionMetric

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection


def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--network', type=str, default='darknet53',
                        help="Base network name")
    parser.add_argument('--algorithm', type=str, default='yolo3',
                        help='YOLO version, default is yolo3')
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Testing dataset.')
    parser.add_argument('--metric', type=str, default='coco',
                        help='Metric to use, either voc or coco.')  # todo vid eval with slow, med, fast, only appl to vid
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='XXXX',
                        help='Saving parameter prefix')
    args = parser.parse_args()
    return args


def yolo3_darknet53(classes, dataset_name, transfer=None, pretrained_base=True, pretrained=False,
                    norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on any dataset. Modified from:
    https://github.com/dmlc/gluon-cv/blob/0dbd05c5eb8537c25b64f0e87c09be979303abf2/gluoncv/model_zoo/yolo/yolo3.py

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    dataset_name : str
        The name of the dataset, used for model save name
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from yolo networks trained on other
        datasets.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    if transfer is None:
        base_net = darknet53(
            pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
        strides = [8, 16, 32]
        net = get_yolov3(
            'darknet53', stages, [512, 256, 128], anchors, strides, classes, dataset_name,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    else:
        net = get_model('yolo3_darknet53_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net


def get_dataset(dataset, metric, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = VOCDetection(root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'), splits=[(2007, 'test')])
    elif dataset.lower() == 'coco':
        val_dataset = COCODetection(root=os.path.join('datasets', 'MSCoco'),
                                    splits='instances_val2017', skip_empty=False)
    elif dataset.lower() == 'det':
        val_dataset = ImageNetDetection(root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
                                        splits=['val'], allow_empty=False)
    elif dataset.lower() == 'vid':
        val_dataset = ImageNetVidDetection(root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
                                           splits=[(2017, 'val')], allow_empty=False, videos=False, frames=25)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))

    if metric == 'voc':
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif metric == 'coco':
        val_metric = COCODetectionMetric(val_dataset, os.path.join(args.save_prefix, 'eval'), cleanup=True,
                                         data_shape=(data_shape, data_shape))
    elif metric == 'vid':
        val_metric = VIDDetectionMetric(val_dataset, iou_thresh=0.5, data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Mertic: {} not implemented.'.format(metric))
    return val_dataset, val_metric


def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers, batchify_fn=batchify_fn,)
    return val_loader


def validate(net, val_data, ctx, classes, size, metric):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    metric.reset()
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    with tqdm(total=size) as pbar:
        for ib, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            pbar.update(batch[0].shape[0])
    return metric.get()


if __name__ == '__main__':
    args = parse_args()

    # testing contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # testing dataset
    val_dataset, val_metric = get_dataset(args.dataset, args.metric, args.data_shape)

    # network
    net_name = '_'.join((args.algorithm, args.network, args.dataset))
    os.makedirs(os.path.join('models', args.save_prefix), exist_ok=True)
    args.save_prefix = os.path.join('models', args.save_prefix, net_name)
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(net_name, root='models', pretrained=True, classes=val_dataset.classes)
    else:
        # net = gcv.model_zoo.get_model(net_name, root='models', pretrained=False, classes=val_dataset.classes)
        net = yolo3_darknet53(val_dataset.classes, args.dataset, root='models', pretrained_base=True)
        net.load_parameters(args.pretrained.strip())

    # testing dataloader
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    names, values = validate(net, val_data, ctx, val_dataset.classes, len(val_dataset), val_metric)
    with open(args.pretrained.strip()[:-7]+'_'+args.metric+'.txt', 'w') as f:
        for k, v in zip(names, values):
            print(k, v)
            f.write('{} {}\n'.format(k, v))
