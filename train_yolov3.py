"""Train YOLOv3 with random shapes."""
import argparse
import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from gluoncv import utils as gutils
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils import LRScheduler, LRSequential

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection

from metrics.pascalvoc import VOCMApMetric
from metrics.mscoco import COCODetectionMetric

from models.definitions import yolo3_darknet53, yolo3_mobilenet1_0

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='darknet53',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... " +
                             "Training is with random shapes from (320 to 608).")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./yolo3_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.'
                             'Set to -1 if using resume as a directory and resume from auto found latest epoch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                        help='epochs at which learning rate decays. default is 160,180.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='XXXX',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=-10,
                        help='Saving parameters epoch interval, best model will always be saved. '
                             'Can enter a negative int to save every 1 epochs, but delete after reach -save_interval')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-random-shape', action='store_true',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                        'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    parser.add_argument('--allow_empty', action='store_true', help='Allow samples that contain 0 boxes as [-1s * 6].')
    parser.add_argument('--frames', type=float, default=0.04,
                        help='Based per video - and is NOT randomly sampled:'
                             'If <1: Percent of the full dataset to take eg. .04 (every 25th frame) - range(0, len(video), int(1/frames))'
                             'If >1: This many frames per video - range(0, len(video), int(ceil(len(video)/frames)))'
                             'If =1: Every sample used - full dataset')
    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = VOCDetection(
            root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = VOCDetection(
            root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
            splits=[(2007, 'test')])
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = COCODetection(
            root=os.path.join('datasets', 'MSCoco'), splits='instances_train2017', use_crowd=False)
        val_dataset = COCODetection(
            root=os.path.join('datasets', 'MSCoco'), splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
    elif dataset.lower() == 'det':
        train_dataset = ImageNetDetection(
            root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
            splits=['train'], allow_empty=args.allow_empty)
        val_dataset = ImageNetDetection(
            root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
            splits=['val'], allow_empty=args.allow_empty)
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'vid':
        train_dataset = ImageNetVidDetection(
            root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
            splits=[(2017, 'train')], allow_empty=args.allow_empty, videos=False, frames=args.frames)
        val_dataset = ImageNetVidDetection(
            root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
            splits=[(2017, 'val')], allow_empty=args.allow_empty, videos=False, frames=args.frames)
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    if args.mixup:
        from gluoncv.data import MixupDetection
        train_dataset = MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
    if args.no_random_shape:
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='discard', num_workers=num_workers)
    # NOTE for val batch loader last_batch='keep' changed to last_batch='discard' so exception not thrown
    # when last batch size is smaller than the number of GPUS (which throws exception) this is fixed in gluon
    # PR 14607: https://github.com/apache/incubator-mxnet/pull/14607 - but yet to be in official release
    # discarding last batch will incur minor changes in val results as some val data wont be processed


    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))

    if save_interval > 0 and epoch % save_interval == 0:  # save only these epochs
        # net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))
        net.save_parameters('{:s}_{:04d}.params'.format(prefix, epoch))

    if save_interval < 0:  # save every epoch, but delete nonwanted when reach a desired interval...
        # good for if training stopped within intervals and dont want to waste space with save_interval = 1
        net.save_parameters('{:s}_{:04d}.params'.format(prefix, epoch))

        if epoch % -save_interval == 0:  # delete the ones we don't want
            st = epoch + save_interval + 1
            for d in range(max(0, st), epoch):
                if os.path.exists('{:s}_{:04d}.params'.format(prefix, d)):
                    os.remove('{:s}_{:04d}.params'.format(prefix, d))


def resume(net, async_net, args):
    """Resume model, can find the latest automatically"""
    # Requires the first digit of epoch in save string is a 0, otherwise may need to reimplement with .split()
    if args.start_epoch == -1:
        files = os.listdir(args.resume.strip())
        files = [file for file in files if '_0' in file]
        files = [file for file in files if '.params' in file]
        files.sort()
        resume_file = files[-1]
        args.start_epoch = int(resume_file[:-7].split('_')[-1]) + 1

        net.load_parameters(os.path.join(args.resume.strip(), resume_file))
        async_net.load_parameters(os.path.join(args.resume.strip(), resume_file))
    else:
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
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
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True

    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - args.warmup_epochs for e in lr_decay_epoch]
    num_batches = args.num_samples // args.batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=args.lr,
                    nepochs=args.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(args.lr_mode, base_lr=args.lr,
                    nepochs=args.epochs - args.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=args.lr_decay, power=2),
    ])

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
        kvstore='local')

    # targets
    sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    l1_loss = gluon.loss.L1Loss()

    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    if args.resume.strip():
        with open(args.save_prefix+'_best_map.log', 'r') as f:
            lines = [line.split()[1] for line in f.readlines()]
            best_map = [float(lines[-1])]
    else:
        best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        if args.mixup:
            # TODO(zhreshold): more elegant way to control mixup during runtime
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                try:
                    train_data._dataset.set_mixup(None)
                except AttributeError:
                    train_data._dataset._data.set_mixup(None)

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)

    # network
    os.makedirs(os.path.join('models', args.save_prefix), exist_ok=bool(args.resume.strip()))
    net_name = '_'.join(('yolo3', args.network, args.dataset))
    args.save_prefix = os.path.join('models', args.save_prefix, net_name)

    if args.network == 'darknet53':
        if args.syncbn and len(ctx) > 1:
            net = yolo3_darknet53(train_dataset.classes, args.dataset, root='models', pretrained_base=True,
                                  norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                  norm_kwargs={'num_devices': len(ctx)})
            async_net = yolo3_darknet53(train_dataset.classes, args.dataset, root='models', pretrained_base=False)  # used by cpu worker
        else:
            net = yolo3_darknet53(train_dataset.classes, args.dataset, root='models', pretrained_base=True)
            async_net = net
    elif args.network == 'mobilenet1_0':
        if args.syncbn and len(ctx) > 1:
            net = yolo3_mobilenet1_0(train_dataset.classes, args.dataset, root='models', pretrained_base=True,
                                     norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                     norm_kwargs={'num_devices': len(ctx)})
            async_net = yolo3_mobilenet1_0(train_dataset.classes, args.dataset, root='models',
                                           pretrained_base=False)  # used by cpu worker
        else:
            net = yolo3_mobilenet1_0(train_dataset.classes, args.dataset, root='models', pretrained_base=True)
            async_net = net
    else:
        raise NotImplementedError('Model: {} not implemented.'.format(args.network))

    if args.resume.strip():
        resume(net, async_net, args)
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()

    train_data, val_data = get_dataloader(
        async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
