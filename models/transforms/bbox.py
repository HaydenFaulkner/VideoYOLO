"""Bounding boxes transformation functions."""
from __future__ import division

import copy
import numpy as np
import random

from utils.bbox import bbox_iou

__all__ = ['random_crop_with_constraints', 'crop', 'flip', 'resize', 'translate']


def random_crop_with_constraints(bboxs, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    """Crop an image randomly with bounding box constraints.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Parameters
    ----------
    bboxs : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.

    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).

    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = size

    td = False
    if not isinstance(bboxs, list):
        bboxs = [bboxs]
        td = True

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            emp = True
            for bbox in bboxs:
                if len(bbox) != 0:
                    emp = False
            if emp:  # all have to be empty, then we will just random crop image
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                if td:
                    bboxs = bboxs[0]
                return bboxs, (left, top, right-left, bottom-top)

            cand = True
            for bbox in bboxs:
                iou = bbox_iou(bbox, crop_bb[np.newaxis])
                if min_iou > iou.min() or iou.max() > max_iou:
                    cand = False

            if cand:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop_b = candidates.pop(np.random.randint(0, len(candidates)))
        new_bboxs = crop(bboxs, crop_b, allow_outside_center=False)
        for new_bbox in new_bboxs:
            if new_bbox.size < 1:  # if any are empty try again?
                continue
        new_crop = (crop_b[0], crop_b[1], crop_b[2], crop_b[3])
        return new_bboxs, new_crop
    return bboxs, (0, 0, w, h)


def crop(bboxs, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.

    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.

    Parameters
    ----------
    bboxs : numpy.ndarray or list
        Numpy.ndarray with shape (N, 4+) list of len T, where N is the number of bounding boxes and T is num timesteps.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.

    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    td = False
    if not isinstance(bboxs, list):
        bboxs = [bboxs]
        td = True

    bboxs = copy.deepcopy(bboxs)
    for bbox in bboxs:

        if crop_box is None:
            break
        if not len(crop_box) == 4:
            raise ValueError(
                "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
        if sum([int(c is None) for c in crop_box]) == 4:
            break

        l, t, w, h = crop_box

        left = l if l else 0
        top = t if t else 0
        right = left + (w if w else np.inf)
        bottom = top + (h if h else np.inf)
        crop_bbox = np.array((left, top, right, bottom))

        if allow_outside_center:
            mask = np.ones(bbox.shape[0], dtype=bool)
        else:
            centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
            mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

        # transform borders
        bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
        bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
        bbox[:, :2] -= crop_bbox[:2]
        bbox[:, 2:4] -= crop_bbox[:2]

        mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
        bbox = bbox[mask]

    if td:
        bboxs = bboxs[0]

    return bboxs


def flip(bboxs, size, flip_x=False, flip_y=False):
    """Flip bounding boxes according to image flipping directions.

    Parameters
    ----------
    bboxs : numpy.ndarray or list
        Numpy.ndarray with shape (N, 4+) list of len T, where N is the number of bounding boxes and T is num timesteps.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2: (width, height).
    flip_x : bool
        Whether flip horizontally.
    flip_y : type
        Whether flip vertically.

    Returns
    -------
    numpy.ndarray
        Flipped bounding boxes with original shape.
    """
    if not len(size) == 2:
        raise ValueError("size requires length 2 tuple, given {}".format(len(size)))

    width, height = size

    td = False
    if not isinstance(bboxs, list):
        bboxs = [bboxs]
        td = True

    bboxs = copy.deepcopy(bboxs)
    for bbox in bboxs:
        if flip_y:
            ymax = height - bbox[:, 1]
            ymin = height - bbox[:, 3]
            bbox[:, 1] = ymin
            bbox[:, 3] = ymax
        if flip_x:
            xmax = width - bbox[:, 0]
            xmin = width - bbox[:, 2]
            bbox[:, 0] = xmin
            bbox[:, 2] = xmax

    if td:
        bboxs = bboxs[0]

    return bboxs


def resize(bboxs, in_size, out_size):
    """Resize bouding boxes according to image resize operation.

    Parameters
    ----------
    bbox : numpy.ndarray or list
        Numpy.ndarray with shape (N, 4+) list of len T, where N is the number of bounding boxes and T is num timesteps.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.

    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    td = False
    if not isinstance(bboxs, list):
        bboxs = [bboxs]
        td = True

    bboxs = copy.deepcopy(bboxs)
    for bbox in bboxs:

        x_scale = out_size[0] / in_size[0]
        y_scale = out_size[1] / in_size[1]
        bbox[:, 1] = y_scale * bbox[:, 1]
        bbox[:, 3] = y_scale * bbox[:, 3]
        bbox[:, 0] = x_scale * bbox[:, 0]
        bbox[:, 2] = x_scale * bbox[:, 2]

    if td:
        bboxs = bboxs[0]

    return bboxs


def translate(bboxs, x_offset=0, y_offset=0):
    """Translate bounding boxes by offsets.

    Parameters
    ----------
    bboxs : numpy.ndarray or list
        Numpy.ndarray with shape (N, 4+) list of len T, where N is the number of bounding boxes and T is num timesteps.
        The last axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    x_offset : int or float
        Offset along x axis.
    y_offset : int or float
        Offset along y axis.

    Returns
    -------
    numpy.ndarray or list
        Translated bounding boxes with original shape.
    """
    td = False
    if not isinstance(bboxs, list):
        bboxs = [bboxs]
        td = True

    bboxs = copy.deepcopy(bboxs)
    for bbox in bboxs:
        bbox[:, :2] += (x_offset, y_offset)
        bbox[:, 2:4] += (x_offset, y_offset)

    if td:
        bboxs = bboxs[0]

    return bboxs
