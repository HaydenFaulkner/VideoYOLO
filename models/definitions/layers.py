"""
File containing the custom layers
"""

from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm


# Functions
def _upsample(x, stride=2):
    """Simple upsampling layer by stack pixel alongside horizontal and vertical directions.
    Parameters
    ----------
    x : mxnet.nd.NDArray or mxnet.symbol.Symbol
        The input array.
    stride : int, default is 2
        Upsampling stride
    """
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)


def _temp_pad(F, x, padding=1, zeros=True):
    """
    Pads a 3D input along temporal axis by repeating edges or zeros
    Args:
        x: dim 5 b,t,c,w,h
        padding: the number of dim to add on each side
        zeros: pad with zeros?

    Returns: padded x

    """
    first = x.slice_axis(axis=1, begin=0, end=1)  # symbol compatible indexing
    last = x.slice_axis(axis=1, begin=-1, end=None)
    if zeros:
        first = first * 0
        last = last * 0
    if padding > 1:
        first = first.repeat(repeats=padding, axis=1)
        last = last.repeat(repeats=padding, axis=1)

    x = F.concat(first, x, dim=1)
    x = F.concat(x, last, dim=1)

    return x


# HybridSequentials
def _conv1d(out_channels, kernel, padding, strides, norm_layer=BatchNorm, norm_kwargs=None):
    """1D over t*c for joining temps"""
    cell = nn.HybridSequential(prefix='1D')

    cell.add(nn.Conv3D(out_channels, kernel_size=(kernel, 1, 1), strides=(strides, 1, 1),
                       padding=(padding, 0, 0), use_bias=False, groups=out_channels, weight_initializer='zeros'))

    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))

    return cell


def _conv2d(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


def _conv3d(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common 3dconv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='3D')
    cell.add(nn.Conv3D(channel, kernel_size=kernel, strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


def _conv21d(channel, t, d, m, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """R(2+1)D from 'A Closer Look at Spatiotemporal Convolutions for Action Recognition'"""
    cell = nn.HybridSequential(prefix='R(2+1)D')

    cell.add(_conv3d(m, (1, d, d), (0, padding[0], padding[0]), stride[0], norm_layer=norm_layer, norm_kwargs=norm_kwargs))
    cell.add(_conv3d(channel, (t, 1, 1), (padding[1], 0, 0), stride[1], norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    return cell


# HybridBlocks
class Corr(gluon.HybridBlock):
    def __init__(self, d, t, kernal_size=1, stride=1, keep='all', comp_mid=False, **kwargs):
        """
        Correlation helper layer, can perform over t time-steps
        """
        super(Corr, self).__init__(**kwargs)

        # used for determining whether to also concat the k features of just the middle with the corr filters
        assert keep in ['all', 'mid', 'none']
        self._keep = keep
        self._d = d
        self._t = t
        self._kernal_size = kernal_size
        self._stride = stride
        self._comp_mid = comp_mid

    def hybrid_forward(self, F, x):
        xs = F.split(x, self._t, axis=1)
        middle_index = int(self._t/2)
        if self._keep == 'all':  # keep all t features
            x = F.reshape(x, (0, -3, -2))
        elif self._keep == 'mid':  # just keep the middle feature
            x = F.squeeze(xs[middle_index], axis=1)

        for i, t in enumerate(xs):  # calculate the correlation features across all t
            if not self._comp_mid and i == middle_index:  # but skip comparing the middle one
                continue
            c = F.Correlation(F.squeeze(t, axis=1), F.squeeze(xs[middle_index], axis=1),
                              kernel_size=self._kernal_size, max_displacement=self._d, pad_size=self._d+int(self._kernal_size/2),
                              stride1=self._stride, stride2=self._stride)
            if self._keep == 'none':
                if i == 0:  # just keep the correlation feats throwing away the x
                    x = F.expand_dims(c, axis=1)
                    continue
                else:
                    c = F.expand_dims(c, axis=1)

            x = F.concat(x, c, dim=1)

        return x


class Conv(gluon.HybridBlock):
    def __init__(self, type, channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        """
        Convolution helper layer, can perform 2d, 3d and 2+1d
        """
        super(Conv, self).__init__(**kwargs)

        assert type in ['2', '3', '21']

        self._type = type

        with self.name_scope():
            if type == '2':
                self.conv = _conv2d(channel=channel, kernel=kernel, padding=padding, stride=stride,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
            elif type == '3':
                self.conv = _conv3d(channel=channel, kernel=kernel, padding=padding, stride=stride,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
            else:
                self.conv = _conv21d(channel=channel, t=kernel, d=kernel, m=channel, padding=[padding, padding],
                                     stride=[stride, stride], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

    def hybrid_forward(self, F, x):
        return self.conv(x)


class TemporalPooling(gluon.HybridBlock):
    def __init__(self, k, type='max', pool_size=None, strides=None, padding=0, style='direct', **kwargs):
        """
        A Temporal Pooling Layer
        """
        super(TemporalPooling, self).__init__(**kwargs)

        assert type in ['max', 'mean']
        assert style in ['direct', 'layer']

        self._type = type
        self._style = style

        if pool_size is None:
            pool_size = k
        else:
            print("Particular pool size specified, so need to use 'layer' style")
            style = 'layer'

        if style == 'layer':
            with self.name_scope():
                if type == 'max':
                    self.pool = gluon.nn.MaxPool1D(pool_size=pool_size,
                                                   strides=strides,
                                                   padding=padding,
                                                   layout='NWC')
                else:
                    self.pool = gluon.nn.AvgPool1D(pool_size=pool_size,
                                                   strides=strides,
                                                   padding=padding,
                                                   layout='NWC')

    def hybrid_forward(self, F, x):
        if self._style == 'layer':
            shp = x
            x = F.reshape(x, (0, 0, -1))
            x = self.pool(x)
            x = F.reshape_like(x, shp, lhs_begin=2, rhs_begin=2)
            x = F.squeeze(x, axis=1)
            return x
        else:
            if self._type == 'max':
                return F.squeeze(F.max(x, axis=1, keepdims=True), axis=1)
            else:
                return F.squeeze(F.mean(x, axis=1, keepdims=True), axis=1)


class TimeDistributed(gluon.HybridBlock):
    def __init__(self, model, style='reshape1', **kwargs):
        """
        A time distributed layer like that seen in Keras
        Args:
            model: the backbone model that will be repeated over time
            style (str): either 'reshape1', 'reshape2' or 'for' for the implementation to use (default is reshape1)
                         NOTE!!: Only reshape1 works with hybrid models
        """
        super(TimeDistributed, self).__init__(**kwargs)
        assert style in ['reshape1', 'reshape2', 'for']

        # if style != 'reshape1':
        #     print("WARNING: net can't be hybridized if {} is used for the TimeDistributed layer style".format(style))

        self._style = style
        with self.name_scope():
            self.model = model

    def apply_model(self, x, _):
        return self.model(x), []

    def hybrid_forward(self, F, x):
        if self._style == 'for':
            # For loop style
            x = F.swapaxes(x, 0, 1)  # swap batch and seqlen channels
            x, _ = F.contrib.foreach(self.apply_model, x, [])  # runs on first channel, which is now seqlen
            if isinstance(x, tuple):  # for handling multiple outputs
                x = (F.swapaxes(xi, 0, 1) for xi in x)
            elif isinstance(x, list):
                x = [F.swapaxes(xi, 0, 1) for xi in x]
            else:
                x = F.swapaxes(x, 0, 1)  # swap seqlen and batch channels
        elif self._style == 'reshape1':
            shp = x  # can use this to keep shapes for reshape back to (batch, timesteps, ...)
            x = F.reshape(x, (-3, -2))  # combines batch and timesteps dims
            x = self.model(x)
            if isinstance(x, tuple):  # for handling multiple outputs
                x = (F.reshape_like(xi, shp, lhs_end=1, rhs_end=2) for xi in x)
            elif isinstance(x, list):
                x = [F.reshape_like(xi, shp, lhs_end=1, rhs_end=2) for xi in x]
            else:
                x = F.reshape_like(x, shp, lhs_end=1, rhs_end=2)  # (num_samples, timesteps, ...)
        else:
            # Reshape style, doesn't work with symbols cause no shape
            batch_size = x.shape[0]
            input_length = x.shape[1]
            x = F.reshape(x, (-3, -2))  # combines batch and timesteps dims
            x = self.model(x)
            if isinstance(x, tuple):  # for handling multiple outputs
                x = (F.reshape(xi, (batch_size, input_length,) + xi.shape[1:]) for xi in x)
            elif isinstance(x, list):
                x = [F.reshape(xi, (batch_size, input_length,) + xi.shape[1:]) for xi in x]
            else:
                x = F.reshape(x, (batch_size, input_length,) + x.shape[1:])  # (num_samples, timesteps, ...)

        return x


class RNN(gluon.HybridBlock):
    def __init__(self, k, input_shape, type='gru', channels=None, kernel=(3,3), bi=True, **kwargs):
        """
        An RNN Layer
        """
        super(RNN, self).__init__(**kwargs)

        assert type in ['gru', 'lstm']

        self._k = k
        self._bi = bi

        with self.name_scope():
            pad = (0,0)
            if kernel[0] == 3:
                pad = (1,1)
            if type == 'gru':
                a = gluon.contrib.rnn.Conv2DGRUCell(input_shape=input_shape, hidden_channels=channels,
                                                    i2h_kernel=kernel, h2h_kernel=kernel, i2h_pad=pad)
                if bi:
                    b = gluon.contrib.rnn.Conv2DGRUCell(input_shape=input_shape, hidden_channels=channels,
                                                        i2h_kernel=kernel, h2h_kernel=kernel, i2h_pad=pad)
            else:
                a = gluon.contrib.rnn.Conv2DLSTMCell(input_shape=input_shape, hidden_channels=channels,
                                                     i2h_kernel=kernel, h2h_kernel=kernel, i2h_pad=pad)
                if bi:
                    b = gluon.contrib.rnn.Conv2DLSTMCell(input_shape=input_shape, hidden_channels=channels,
                                                         i2h_kernel=kernel, h2h_kernel=kernel, i2h_pad=pad)

            if bi:
                self.rnn = gluon.rnn.BidirectionalCell(a, b)
            else:
                self.rnn = a

    def hybrid_forward(self, F, x):
        x, h = self.rnn.unroll(self._k, x, merge_outputs=True)
        if self._bi:
            x = F.split(x, 2, axis=2)  # avg the two bi dir channels
            x = (x[0] + x[1]) / 2
        return x