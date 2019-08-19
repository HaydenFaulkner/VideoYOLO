from __future__ import division
from __future__ import print_function

from absl import app, flags, logging
from absl.flags import FLAGS
import os
import logging
import time
import warnings
import numpy as np

from gluoncv import utils as gutils
from gluoncv.data.batchify import Tuple, Stack, Pad
# from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform, YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils import LRScheduler, LRSequential
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from tensorboardX import SummaryWriter

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection

from metrics.pascalvoc import VOCMApMetric
from metrics.mscoco import COCODetectionMetric

from models.definitions import yolo3_darknet53, yolo3_mobilenet1_0
from models.transforms import YOLO3DefaultTrainTransform, YOLO3DefaultInferenceTransform

from utils.general import as_numpy

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

logging.basicConfig(level=logging.INFO)

flags.DEFINE_string('network', 'darknet53',
                    'Base network name: darknet53 or mobilenet1.0.')
flags.DEFINE_string('dataset', 'voc',
                    'Dataset to train on.')
flags.DEFINE_string('trained_on', '',
                    'Used for finetuning, specify the dataset the original model was trained on.')
flags.DEFINE_string('save_prefix', '0001',
                    'Model save prefix.')
flags.DEFINE_integer('log_interval', 100,
                     'Logging mini-batch interval.')
flags.DEFINE_integer('save_interval', -10,
                     'Saving parameters epoch interval, best model will always be saved. '
                     'Can enter a negative int to save every 1 epochs, but delete after reach -save_interval')
flags.DEFINE_integer('val_interval', 1,
                     'Epoch interval for validation.')
flags.DEFINE_string('resume', '',
                    'Resume from previously saved parameters if not None.')

flags.DEFINE_integer('batch_size', 64,
                     'Batch size for detection: higher faster, but more memory intensive.')
flags.DEFINE_integer('epochs', 200,
                     'How many training epochs to complete')
flags.DEFINE_integer('start_epoch', 0,
                     'Starting epoch for resuming, default is 0 for new training.'
                     'You can specify it to 100 for example to start from 100 epoch.'
                     'Set to -1 if using resume as a directory and resume from auto found latest epoch')
flags.DEFINE_integer('data_shape', 416,
                     'For evaluation, use 320, 416, 608... Training is with random shapes from (320 to 608).')
flags.DEFINE_float('lr', 0.001,
                   'Learning rate.')
flags.DEFINE_string('lr_mode', 'step',
                    'Learning rate scheduler mode. options are step, poly and cosine.')
flags.DEFINE_float('lr_decay', 0.1,
                   'Decay rate of learning rate.')
flags.DEFINE_integer('lr_decay_period', 0,
                     'Interval for periodic learning rate decays.')
flags.DEFINE_list('lr_decay_epoch', [160, 180],
                  'Epochs at which learning rate decays.')
flags.DEFINE_float('warmup_lr', 0.0,
                   'Starting warmup learning rate.')
flags.DEFINE_integer('warmup_epochs', 0,
                     'Number of warmup epochs.')
flags.DEFINE_float('momentum', 0.9,
                   'SGD momentum.')
flags.DEFINE_float('wd', 0.0005,
                   'Weight decay.')

flags.DEFINE_boolean('pretrained_cnn', True,
                     'Use an imagenet pretrained cnn as base network.')
flags.DEFINE_boolean('syncbn', False,
                     'Use synchronize BN across devices.')
flags.DEFINE_boolean('no_random_shape', False,
                     'Use fixed size(data-shape) throughout the training, which will be faster '
                     'and require less memory. However, final model will be slightly worse.')
flags.DEFINE_boolean('no_wd', False,
                     'Remove weight decay on bias, and beta/gamma for batchnorm layers.')
flags.DEFINE_boolean('mixup', False,
                     'Enable mixup?')
flags.DEFINE_integer('no_mixup_epochs', 20,
                     'Disable mixup training if enabled in the last N epochs.')
flags.DEFINE_boolean('label_smooth', False,
                     'Use label smoothing?')
flags.DEFINE_boolean('allow_empty', False,
                     'Allow samples that contain 0 boxes as [-1s * 6]?')

flags.DEFINE_list('gpus', [0],
                  'GPU IDs to use. Use comma for multiple eg. 0,1.')
flags.DEFINE_integer('num_workers', 8,
                     'The number of workers should be picked so that it’s equal to number of cores on your machine '
                     'for max parallelization. If this number is bigger than your number of cores it will use up '
                     'a bunch of extra CPU memory.')

flags.DEFINE_integer('num_samples', -1,
                     'Training images. Use -1 to automatically get the number.')
flags.DEFINE_float('frames', 0.04,
                   'Based per video - and is NOT randomly sampled:'
                   'If <1: Percent of the full dataset to take eg. .04 (every 25th frame) - range(0, len(video), int(1/frames))'
                   'If >1: This many frames per video - range(0, len(video), int(ceil(len(video)/frames)))'
                   'If =1: Every sample used - full dataset')
flags.DEFINE_integer('seed', 233,
                     'Random seed to be fixed.')


def get_dataset(dataset_name, save_prefix=''):
    if dataset_name.lower() == 'voc':
        train_dataset = VOCDetection(
            root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = VOCDetection(
            root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
            splits=[(2007, 'test')])
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)

    elif dataset_name.lower() == 'coco':
        train_dataset = COCODetection(
            root=os.path.join('datasets', 'MSCoco'), splits='instances_train2017', use_crowd=False)
        val_dataset = COCODetection(
            root=os.path.join('datasets', 'MSCoco'), splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, save_prefix + '_eval', cleanup=True,
            data_shape=(FLAGS.data_shape, FLAGS.data_shape))

    elif dataset_name.lower() == 'det':
        train_dataset = ImageNetDetection(
            root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
            splits=['train'], allow_empty=FLAGS.allow_empty)
        val_dataset = ImageNetDetection(
            root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
            splits=['val'], allow_empty=FLAGS.allow_empty)
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)

    elif dataset_name.lower() == 'vid':
        train_dataset = ImageNetVidDetection(
            root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
            splits=[(2017, 'train')], allow_empty=FLAGS.allow_empty, videos=False, frames=FLAGS.frames)
        val_dataset = ImageNetVidDetection(
            root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
            splits=[(2017, 'val')], allow_empty=FLAGS.allow_empty, videos=False, frames=FLAGS.frames)
        val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)

    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset_name))

    if FLAGS.mixup:
        from gluoncv.data import MixupDetection
        train_dataset = MixupDetection(train_dataset)

    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, batch_size):
    """Get dataloader."""
    width, height = FLAGS.data_shape, FLAGS.data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
    if FLAGS.no_random_shape:
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, 3, net, mixup=FLAGS.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=FLAGS.num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, 3, net, mixup=FLAGS.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=FLAGS.num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultInferenceTransform(width, height, 3)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='discard', num_workers=FLAGS.num_workers)
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


def resume(net, async_net, resume, start_epoch):
    """Resume model, can find the latest automatically"""
    # Requires the first digit of epoch in save string is a 0, otherwise may need to reimplement with .split()
    if start_epoch == -1:
        files = os.listdir(resume.strip())
        files = [file for file in files if '_0' in file]
        files = [file for file in files if '.params' in file]
        files.sort()
        resume_file = files[-1]
        start_epoch = int(resume_file[:-7].split('_')[-1]) + 1

        net.load_parameters(os.path.join(resume.strip(), resume_file))
        async_net.load_parameters(os.path.join(resume.strip(), resume_file))
    else:
        net.load_parameters(resume.strip())
        async_net.load_parameters(resume.strip())

    return start_epoch


def get_net(trained_on_dataset, ctx):
    if FLAGS.network == 'darknet53':
        if FLAGS.syncbn and len(ctx) > 1:
            net = yolo3_darknet53(trained_on_dataset.classes, FLAGS.dataset,
                                  root='models',
                                  pretrained_base=FLAGS.pretrained_cnn,
                                  norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                  norm_kwargs={'num_devices': len(ctx)})
            async_net = yolo3_darknet53(trained_on_dataset.classes, FLAGS.dataset,
                                        root='models',
                                        pretrained_base=False)  # used by cpu worker
        else:
            net = yolo3_darknet53(trained_on_dataset.classes, FLAGS.dataset,
                                  root='models',
                                  pretrained_base=FLAGS.pretrained_cnn)
            async_net = net
    elif FLAGS.network == 'mobilenet1.0':
        if FLAGS.syncbn and len(ctx) > 1:
            net = yolo3_mobilenet1_0(trained_on_dataset.classes, FLAGS.dataset,
                                     root='models',
                                     pretrained_base=FLAGS.pretrained_cnn,
                                     norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                     norm_kwargs={'num_devices': len(ctx)})
            async_net = yolo3_mobilenet1_0(trained_on_dataset.classes, FLAGS.dataset,
                                           root='models',
                                           pretrained_base=False)  # used by cpu worker
        else:
            net = yolo3_mobilenet1_0(trained_on_dataset.classes, FLAGS.dataset,
                                     root='models',
                                     pretrained_base=FLAGS.pretrained_cnn)
            async_net = net
    else:
        raise NotImplementedError('Model: {} not implemented.'.format(FLAGS.network))

    if FLAGS.resume.strip():
        start_epoch = resume(net, async_net, FLAGS.resume, FLAGS.start_epoch)
    else:
        start_epoch = FLAGS.start_epoch
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()

    return net, async_net, start_epoch

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

        # lists with results ran on each gpu (ie len of list is = num gpus) in each list is (BatchSize, Data
        # update metric
        # eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        # lodged issue on github #872 https://github.com/dmlc/gluon-cv/issues/872
        eval_metric.update(as_numpy(det_bboxes), as_numpy(det_ids), as_numpy(det_scores), as_numpy(gt_bboxes), as_numpy(gt_ids), as_numpy(gt_difficults))
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, save_prefix, start_epoch, num_samples):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if FLAGS.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if FLAGS.label_smooth:
        net._target_generator._label_smooth = True

    if FLAGS.lr_decay_period > 0:
        lr_decay_epoch = list(range(FLAGS.lr_decay_period, FLAGS.epochs, FLAGS.lr_decay_period))
    else:
        lr_decay_epoch = FLAGS.lr_decay_epoch
    lr_decay_epoch = [e - FLAGS.warmup_epochs for e in lr_decay_epoch]
    num_batches = num_samples // FLAGS.batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=FLAGS.lr,
                    nepochs=FLAGS.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(FLAGS.lr_mode, base_lr=FLAGS.lr,
                    nepochs=FLAGS.epochs - FLAGS.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=FLAGS.lr_decay, power=2),
    ])

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': FLAGS.wd, 'momentum': FLAGS.momentum, 'lr_scheduler': lr_scheduler},
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
    log_file_path = save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    # logger.info(FLAGS)

    # set up tensorboard summary writer
    tb_sw = SummaryWriter(log_dir=os.path.join(log_dir, 'tb'), comment=FLAGS.save_prefix)

    # Check if wanting to resume
    logger.info('Start training from [Epoch {}]'.format(start_epoch))
    if FLAGS.resume.strip() and os.path.exists(save_prefix+'_best_map.log'):
        with open(save_prefix+'_best_map.log', 'r') as f:
            lines = [line.split()[1] for line in f.readlines()]
            best_map = [float(lines[-1])]
    else:
        best_map = [0]

    # Training loop
    for epoch in range(start_epoch, FLAGS.epochs+1):
        if FLAGS.mixup:
            # TODO(zhreshold): more elegant way to control mixup during runtime
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= FLAGS.epochs - FLAGS.no_mixup_epochs:
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
            if FLAGS.log_interval and not (i + 1) % FLAGS.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                tb_sw.add_scalar(tag='Training_' + name1, scalar_value=loss1, global_step=(epoch * len(train_data) + i))
                tb_sw.add_scalar(tag='Training_' + name2, scalar_value=loss2, global_step=(epoch * len(train_data) + i))
                tb_sw.add_scalar(tag='Training_' + name3, scalar_value=loss3, global_step=(epoch * len(train_data) + i))
                tb_sw.add_scalar(tag='Training_' + name4, scalar_value=loss4, global_step=(epoch * len(train_data) + i))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if not (epoch + 1) % FLAGS.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            tb_sw.add_scalar(tag='Validation_mAP', scalar_value=float(mean_ap[-1]),
                             global_step=(epoch * len(train_data) + i))
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, FLAGS.save_interval, save_prefix)


def main(_argv):

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(FLAGS.seed)

    # training contexts
    ctx = [mx.gpu(i) for i in FLAGS.gpus]
    ctx = ctx if ctx else [mx.cpu()]

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(FLAGS.dataset, os.path.join('models', FLAGS.save_prefix))

    trained_on_dataset = train_dataset
    if FLAGS.trained_on:
        # load the model with these classes then reset
        trained_on_dataset, _, _ = get_dataset(FLAGS.trained_on, os.path.join('models', FLAGS.save_prefix))

    # network
    if os.path.exists(os.path.join('models', FLAGS.save_prefix)) and not bool(FLAGS.resume.strip()):
        logging.error("{} exists so won't overwrite and restart training. You can resume training by using "
                      "--resume".format(os.path.join('models', FLAGS.save_prefix)))
        return
    os.makedirs(os.path.join('models', FLAGS.save_prefix), exist_ok=bool(FLAGS.resume.strip()))
    net_name = '_'.join(('yolo3', FLAGS.network, FLAGS.dataset))
    save_prefix = os.path.join('models', FLAGS.save_prefix, net_name)

    net, async_net, start_epoch = get_net(trained_on_dataset, ctx)

    if FLAGS.trained_on:
        net.reset_class(train_dataset.classes)

    # load the dataloader
    train_data, val_data = get_dataloader(async_net, train_dataset, val_dataset, FLAGS.batch_size)

    num_samples = FLAGS.num_samples
    if num_samples < 0:
        num_samples = len(train_dataset)

    # training
    train(net, train_data, val_data, eval_metric, ctx, save_prefix, start_epoch, num_samples)


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass

