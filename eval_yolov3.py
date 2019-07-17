from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
logging.basicConfig(level=logging.INFO)
import mxnet as mx
from tqdm import tqdm
from mxnet import gluon
import gluoncv as gcv
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection

from metrics.pascalvoc import VOCMApMetric
from metrics.mscoco import COCODetectionMetric
from metrics.imgnetvid import VIDDetectionMetric

from models.definitions import yolo3_darknet53, yolo3_mobilenet1_0

from utils.general import as_numpy

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
    parser.add_argument('--trained-on', type=str, default='',
                        help='Dataset that the model was trained on - used to get n_classes.')
    parser.add_argument('--metric', type=str, default='voc',
                        help='Metric to use, either voc or coco.')  # todo vid eval with slow, med, fast, only appl to vid
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='XXXX',
                        help='Saving parameter prefix')
    parser.add_argument('--frames', type=float, default=0.04,
                        help='Based per video - and is NOT randomly sampled:'
                             'If <1: Percent of the full dataset to take eg. .04 (every 25th frame) - range(0, len(video), int(1/frames))'
                             'If >1: This many frames per video - range(0, len(video), int(ceil(len(video)/frames)))'
                             'If =1: Every sample used - full dataset')
    args = parser.parse_args()
    return args


def get_dataset(dataset):
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
                                           splits=[(2017, 'val')], allow_empty=False, videos=False, frames=args.frames)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset


def get_metric(val_dataset, metric, data_shape, class_map=None):
    if metric.lower() == 'voc':
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes, class_map=class_map)
    elif metric.lower() == 'coco':
        val_metric = COCODetectionMetric(val_dataset, os.path.join(args.save_prefix, 'eval'), cleanup=True,
                                         data_shape=(data_shape, data_shape))
    elif metric.lower() == 'vid':
        val_metric = VIDDetectionMetric(val_dataset, iou_thresh=0.5, data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Mertic: {} not implemented.'.format(metric))
    return val_metric


def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers, batchify_fn=batchify_fn,)
    return val_loader


def validate(net, val_data, ctx, size, metric):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    with tqdm(total=size) as pbar:
        for batch in val_data:
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

            # metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            # lodged issue on github #872 https://github.com/dmlc/gluon-cv/issues/872
            metric.update(as_numpy(det_bboxes), as_numpy(det_ids), as_numpy(det_scores), as_numpy(gt_bboxes), as_numpy(gt_ids), as_numpy(gt_difficults))
            pbar.update(batch[0].shape[0])
    return metric.get()


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


if __name__ == '__main__':
    args = parse_args()

    # testing contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # testing dataset
    val_dataset = get_dataset(args.dataset)

    if args.trained_on: # for use when model preds are diff to eval set classes
        trained_on_dataset = get_dataset(args.trained_on)
        val_metric = get_metric(val_dataset, args.metric, args.data_shape, class_map=get_class_map(trained_on_dataset,
                                                                                                   val_dataset))
    else:
        trained_on_dataset = val_dataset
        val_metric = get_metric(val_dataset, args.metric, args.data_shape)

    # network
    net_name = '_'.join((args.algorithm, args.network, args.dataset))
    os.makedirs(os.path.join('models', args.save_prefix), exist_ok=True)
    args.save_prefix = os.path.join('models', args.save_prefix, net_name)
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(net_name, root='models', pretrained=True, classes=trained_on_dataset.classes)
    else:
        if args.network == 'darknet53':
            if len(ctx) > 1:
                net = yolo3_darknet53(trained_on_dataset.classes, args.dataset, root='models', pretrained_base=True,
                                      norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                      norm_kwargs={'num_devices': len(ctx)})
                async_net = yolo3_darknet53(trained_on_dataset.classes, args.dataset, root='models',
                                            pretrained_base=False)  # used by cpu worker
            else:
                net = yolo3_darknet53(trained_on_dataset.classes, args.dataset, root='models', pretrained_base=True)
                async_net = net
        elif args.network == 'mobilenet1_0':
            if len(ctx) > 1:
                net = yolo3_mobilenet1_0(trained_on_dataset.classes, args.dataset, root='models', pretrained_base=True,
                                         norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                         norm_kwargs={'num_devices': len(ctx)})
                async_net = yolo3_mobilenet1_0(trained_on_dataset.classes, args.dataset, root='models',
                                               pretrained_base=False)  # used by cpu worker
            else:
                net = yolo3_mobilenet1_0(trained_on_dataset.classes, args.dataset, root='models', pretrained_base=True)
                async_net = net
        else:
            raise NotImplementedError('Model: {} not implemented.'.format(args.network))

        net.load_parameters(args.pretrained.strip())

    # testing dataloader
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    names, values = validate(net, val_data, ctx, len(val_dataset), val_metric)
    with open(args.pretrained.strip()[:-7]+'_D'+args.dataset+'_M'+args.metric+'.txt', 'w') as f:
        for k, v in zip(names, values):
            print(k, v)
            f.write('{} {}\n'.format(k, v))
