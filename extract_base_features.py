from __future__ import division
from __future__ import print_function

from absl import app, flags, logging
from absl.flags import FLAGS
import glob
from gluoncv.model_zoo.yolo.darknet import darknet53
from gluoncv.model_zoo.mobilenet import get_mobilenet
from gluoncv.data.batchify import Tuple, Stack, Pad
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.nn import BatchNorm
import logging
import numpy as np
import os
from tqdm import tqdm

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection
from datasets.detectset import DetectSet

from models.definitions.yolo.transforms import YOLO3VideoInferenceTransform

from utils.general import as_numpy
from utils.video import video_to_frames

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

logging.basicConfig(level=logging.INFO)

flags.DEFINE_string('network', 'darknet53',
                    'Base network name: darknet53 or mobilenet1.0.')
flags.DEFINE_string('dataset', 'voc',
                    'Dataset or .jpg image or .mp4 video or .txt image/video list.')
flags.DEFINE_string('save_dir', 'features',
                    'Save directory to save images.')
flags.DEFINE_integer('batch_size', 1,
                     'Batch size for detection: higher faster, but more memory intensive.')
flags.DEFINE_integer('data_shape', 416,
                     'Input data shape.')
flags.DEFINE_float('frames', 0.04,
                   'Based per video - and is NOT randomly sampled:'
                   'If <1: Percent of the full dataset to take eg. .04 (every 25th frame) - range(0, len(video), int(1/frames))'
                   'If >1: This many frames per video - range(0, len(video), int(ceil(len(video)/frames)))'
                   'If =1: Every sample used - full dataset')

flags.DEFINE_list('gpus', [0],
                  'GPU IDs to use. Use comma or space for multiple eg. 0,1 or 0 1.')
flags.DEFINE_integer('num_workers', 8,
                     'The number of workers should be picked so that it’s equal to number of cores on your machine'
                     ' for max parallelization.')


def get_dataset(dataset_name):
    if dataset_name.lower() == 'voc':
        # dataset = VOCDetection(splits=[(2007, 'test')], inference=True)
        dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')], inference=True)

    elif dataset_name.lower() == 'coco':
        # dataset = COCODetection(splits=['instances_val2017'], allow_empty=True, inference=True)
        dataset = COCODetection(splits=['instances_train2017'], use_crowd=False, inference=True)

    elif dataset_name.lower() == 'det':
        # dataset = ImageNetDetection(splits=['val'], allow_empty=True, inference=True)
        dataset = ImageNetDetection(splits=['train'], allow_empty=True, inference=True)

    elif dataset_name.lower() == 'vid':
        # dataset = ImageNetVidDetection(splits=[(2017, 'val')], allow_empty=True, frames=FLAGS.frames, inference=True)
        dataset = ImageNetVidDetection(splits=[(2017, 'train')], allow_empty=True, frames=FLAGS.frames,
                                       inference=True)

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


def extract(save_dir, net, dataset, loader, ctx, net_name='darknet53'):
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    with tqdm(total=len(dataset), desc="Extracting") as pbar:
        for ib, batch in enumerate(loader):

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            idxs = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)

            f1s = []
            f2s = []
            f3s = []
            sidxs = []
            for x, y, sidx in zip(data, label, idxs):
                if net_name == 'darknet53':  # darknet
                    f1 = net.features[:15](x)
                    f2 = net.features[15:24](f1)
                    f3 = net.features[24:](f2)
                else:  # mobilenet
                    f1 = net.features[:33](x)
                    f2 = net.features[33:69](f1)
                    f3 = net.features[69:-2](f2)
                f1s.append(f1.asnumpy())
                f2s.append(f2.asnumpy())
                f3s.append(f3.asnumpy())
                sidxs.append(sidx.asnumpy())

            for f1, f2, f3, sidx in zip(f1s, f2s, f3s, sidxs):

                for i, idx in enumerate(sidx):
                    img_path = dataset.sample_path(int(idx))

                    file_id = img_path.split(os.sep)[-1][:-4]
                    if FLAGS.dataset == 'vid':
                        file_id = os.path.join(img_path.split(os.sep)[-2], img_path.split(os.sep)[-1][:-5])
                        os.makedirs(os.path.join(save_dir, img_path.split(os.sep)[-2]), exist_ok=True)

                    np.save(os.path.join(save_dir, file_id + '_F1.npy'), f1[i])
                    np.save(os.path.join(save_dir, file_id + '_F2.npy'), f2[i])
                    np.save(os.path.join(save_dir, file_id + '_F3.npy'), f3[i])

            pbar.update(batch[0].shape[0])

    return None


def main(_argv):

    # get dataset
    dataset = get_dataset(FLAGS.dataset)

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
    if FLAGS.network == 'darknet53':
        net = darknet53(root='models', pretrained=True, norm_layer=BatchNorm, norm_kwargs=None)
    elif FLAGS.network == 'mobilenet1.0':
        net = get_mobilenet(root='models', multiplier=1, pretrained=True, norm_layer=BatchNorm, norm_kwargs=None)
    else:
        raise NotImplementedError('Backbone CNN model {} not implemented.'.format(FLAGS.network))

    # organise the save directories for the results
    if FLAGS.dataset in ['voc', 'coco', 'det', 'vid']:
        save_dir = os.path.join('models', FLAGS.network, FLAGS.save_dir, FLAGS.dataset)
    else:
        save_dir = os.path.join('models', FLAGS.network, FLAGS.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # attempt to load predictions
    extract(save_dir, net, dataset, loader, ctx, net_name=FLAGS.network)


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass
