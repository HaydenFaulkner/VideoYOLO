"""Extended image transformations to `mxnet.image`."""
from __future__ import division
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.base import numeric_types

__all__ = ['random_expand']


def random_expand(src, max_ratio=4, fill=0, keep_ratio=True):
    """Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.

    Modified for video from gluoncv default image transform

    Parameters
    ----------
    src : mxnet.nd.NDArray
        The original image with KHWC format.
    max_ratio : int or float
        Maximum ratio of the output image on both direction(vertical and horizontal)
    fill : int or float or array-like
        The value(s) for padded borders. If `fill` is numerical type, RGB channels
        will be padded with single value. Otherwise `fill` must have same length
        as image channels, which resulted in padding with per-channel values.
    keep_ratio : bool
        If `True`, will keep output image the same aspect ratio as input.

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (offset_x, offset_y, new_width, new_height)

    """
    if max_ratio <= 1:
        return src, (0, 0, src.shape[1], src.shape[0])

    k, h, w, c = src.shape

    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    # make canvas
    if isinstance(fill, numeric_types):
        dst = nd.full(shape=(k, oh, ow, c), val=fill, dtype=src.dtype)
    else:
        fill = nd.array(fill, dtype=src.dtype, ctx=src.context)
        if not c == fill.size:
            raise ValueError("Channel and fill size mismatch, {} vs {}".format(c, fill.size))
        dst = nd.tile(fill.reshape((1, c)), reps=(k * oh * ow, 1)).reshape((k, oh, ow, c))

    dst[:, off_y:off_y+h, off_x:off_x+w, :] = src

    return dst, (off_x, off_y, ow, oh)
