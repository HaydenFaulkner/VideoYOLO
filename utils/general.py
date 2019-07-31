import numpy as np
import mxnet as mx

from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import bbox as tbbox


def as_numpy(a):
    """Convert a (list of) mx.NDArray into numpy.ndarray"""
    if isinstance(a, (list, tuple)):
        out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
        try:
            out = np.concatenate(out, axis=0)
        except ValueError:
            out = np.array(out)
        return out
    elif isinstance(a, mx.nd.NDArray):
        a = a.asnumpy()
    return a
