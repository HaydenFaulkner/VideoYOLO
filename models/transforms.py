"""Transforms for YOLO series."""
from __future__ import absolute_import
import copy
import numpy as np
import mxnet as mx
from mxnet import autograd
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import experimental


class YOLO3DefaultTrainTransform(object):
    """Default YOLO training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    net : mxnet.gluon.HybridBlock, optional
        The yolo network.

        .. hint::

            If net is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, **kwargs):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._mixup = mixup
        self._target_generator = None
        if net is None:
            return

        # in case network has reset_ctx to gpu
        self._fake_x = mx.nd.zeros((1, 3, height, width))  # todo hardcode
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(self._fake_x)
        from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
        self._target_generator = YOLOV3PrefetchTargetGenerator(
            num_class=len(net.classes), **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        if src.shape[2] > 3:  # more than 3 channels
            assert src.shape[2] % 3 == 0
        
        imgs = None
        for still_i in range(int(src.shape[2] / 3)):
            
            # random color jittering
            img = experimental.image.random_color_distort(src[:, :, still_i*3:(still_i*3+3)])
    
            # random expansion with prob 0.5
            if np.random.uniform(0, 1) > 0.5:
                img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
                bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
            else:
                img, bbox = img, label
    
            # random cropping
            h, w, _ = img.shape
            bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
            x0, y0, w, h = crop
            img = mx.image.fixed_crop(img, x0, y0, w, h)
    
            # resize with random interpolation
            h, w, _ = img.shape
            interp = np.random.randint(0, 5)
            img = timage.imresize(img, self._width, self._height, interp=interp)
            bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
    
            # random horizontal flip
            h, w, _ = img.shape
            img, flips = timage.random_flip(img, px=0.5)
            bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
    
            # to tensor
            img = mx.nd.image.to_tensor(img)
            img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

            if imgs is None:
                imgs = img
            else:
                imgs = mx.ndarray.concatenate([imgs, img], axis=0)
                
        img = imgs

        if self._target_generator is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        if self._mixup:
            gt_mixratio = mx.nd.array(bbox[np.newaxis, :, -1:])
        else:
            gt_mixratio = None
        objectness, center_targets, scale_targets, weights, class_targets = self._target_generator(
            self._fake_x, self._feat_maps, self._anchors, self._offsets,
            gt_bboxes, gt_ids, gt_mixratio)
        return (img, objectness[0], center_targets[0], scale_targets[0], weights[0],
                class_targets[0], gt_bboxes[0])


class YOLO3DefaultValTransform(object):
    """Default YOLO validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        h, w, c = src.shape
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        imgs = None
        for still_i in range(int(src.shape[2] / 3)):
            img = timage.imresize(src[:, :, still_i * 3:(still_i * 3 + 3)], self._width, self._height, interp=9)
    
            img = mx.nd.image.to_tensor(img)
            img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

            if imgs is None:
                imgs = img
            else:
                imgs = mx.ndarray.concatenate([imgs, img], axis=2)
    
        img = imgs
        return img, bbox.astype(img.dtype)
