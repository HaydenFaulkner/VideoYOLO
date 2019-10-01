"""Transforms for YOLO series."""
from __future__ import absolute_import
import copy
import numpy as np
import mxnet as mx
from mxnet import autograd
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import experimental

from ..transforms import video as tvideo

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
        self._fake_x = mx.nd.zeros((1, 3, height, width))
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(self._fake_x)
        from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
        self._target_generator = YOLOV3PrefetchTargetGenerator(
            num_class=len(net.classes), **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)

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


class YOLO3DefaultInferenceTransform(object):
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

    def __call__(self, src, label, idx=None):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        if idx is not None:
            return img, bbox.astype(img.dtype), idx
        return img, bbox.astype(img.dtype)


class YOLO3VideoTrainTransform(object):
    """Video YOLO training transform which includes tons of image augmentations.

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
    def __init__(self, k, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, **kwargs):
        self._k = k
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._mixup = mixup
        self._target_generator = None
        if net is None:
            return

        # in case network has reset_ctx to gpu
        if k > 1:
            self._fake_x = mx.nd.zeros((1, k, 3, height, width))
        else:
            self._fake_x = mx.nd.zeros((1, 3, height, width))
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(self._fake_x)

        self._fake_x = mx.nd.zeros((1, 3, height, width))
        from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
        self._target_generator = YOLOV3PrefetchTargetGenerator(num_class=len(net.classes), **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""

        was_three = False
        if len(src.shape) == 3:
            src = mx.nd.expand_dims(src, axis=0)
            was_three = True

        # img=src
        bbox=label
        # random color jittering
        img = experimental.image.random_color_distort(src)  # works for video without modification

        # random expansion with prob 0.5
        # if np.random.uniform(0, 1) > 0.5:
        #     img, expand = tvideo.random_expand(img, fill=[m * 255 for m in self._mean])
        #     bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        # else:
        #     img, bbox = img, label

        # random cropping
        # k, h, w, c = img.shape
        # bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        # x0, y0, w, h = crop
        # img = img[:, y0:y0+h, x0:x0+w, :]

        # resize with random interpolation
        k, h, w, c = img.shape
        interp = np.random.randint(0, 5)
        tmp = mx.nd.ones((k, self._height,  self._width, c), ctx=img.context)
        for i in range(k):
            tmp[i] = timage.imresize(img[i], self._width, self._height, interp=interp)
        img = tmp
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip with prob 0.5
        k, h, w, c = img.shape
        if np.random.uniform(0, 1) > 0.5:
            img = mx.nd.flip(img, axis=1)
            bbox = tbbox.flip(bbox, (w, h), flip_x=True)

        img = mx.nd.image.to_tensor(img)  # to tensor, also transforms from k,h,w,c to k,c,h,w
        # normalise
        for i in range(k):
            img[i] = mx.nd.image.normalize(img[i], mean=self._mean, std=self._std)  # normalise

        if was_three:  # remove the k dimension so backwards compat with single frame
            img = mx.nd.squeeze(img)
            
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


class YOLO3VideoInferenceTransform(object):
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

    def __call__(self, src, label, idx=None):
        """Apply transform to validation image/label."""
        was_three = False
        if len(src.shape) == 3:
            src = mx.nd.expand_dims(src, axis=0)
            was_three = True
            
        # resize
        k, h, w, c = src.shape
        tmp = mx.nd.ones((k, self._height, self._width, c), ctx=src.context)
        for i in range(k):
            tmp[i] = timage.imresize(src[i], self._width, self._height, interp=9)
        img = tmp
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = mx.nd.image.to_tensor(img)  # to tensor, also transforms from k,h,w,c to k,c,h,w
        # normalise
        for i in range(k):
            img[i] = mx.nd.image.normalize(img[i], mean=self._mean, std=self._std)  # normalise

        if was_three:  # remove the k dimension so backwards compat with single frame
            img = mx.nd.squeeze(img)
            
        if idx is not None:
            return img, bbox.astype(img.dtype), idx
        return img, bbox.astype(img.dtype)


class YOLO3NBVideoTrainTransform(object):
    """Video YOLO training transform which includes tons of image augmentations.

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
    def __init__(self, k, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, **kwargs):
        self._k = k
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._mixup = mixup
        self._target_generator = None
        if net is None:
            return

        # in case network has reset_ctx to gpu
        self._fake_x = (mx.nd.zeros((1, 256, int(height / 8), int(width / 8))),
                        mx.nd.zeros((1, 512, int(height / 16), int(width / 16))),
                        mx.nd.zeros((1, 1024, int(height / 32), int(width / 32))))
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(*self._fake_x)
        from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
        self._target_generator = YOLOV3PrefetchTargetGenerator(num_class=len(net.classes), **kwargs)

    def __call__(self, img, f1, f2, f3, bbox):
        """Apply transform to training image/label."""

        if len(img.shape) == 3:
            img = mx.nd.expand_dims(img, axis=0)

        k, h, w, c = img.shape
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
        
        if self._target_generator is None:
            return f1, f2, f3, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        if self._mixup:
            gt_mixratio = mx.nd.array(bbox[np.newaxis, :, -1:])
        else:
            gt_mixratio = None
        objectness, center_targets, scale_targets, weights, class_targets = self._target_generator(
            mx.nd.zeros((1, 3, self._height, self._width)), self._feat_maps, self._anchors, self._offsets,
            gt_bboxes, gt_ids, gt_mixratio)
        return (f1, f2, f3, objectness[0], center_targets[0], scale_targets[0], weights[0],
                class_targets[0], gt_bboxes[0])


class YOLO3NBVideoInferenceTransform(object):
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

    def __call__(self, img, f1, f2, f3, bbox, idx=None):
        """Apply transform to validation image/label."""

        if len(img.shape) == 3:
            img = mx.nd.expand_dims(img, axis=0)

        # resize box
        k, h, w, c = img.shape
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
            
        if idx is not None:
            return f1, f2, f3, bbox.astype(img.dtype), idx
        return f1, f2, f3, bbox.astype(img.dtype)
