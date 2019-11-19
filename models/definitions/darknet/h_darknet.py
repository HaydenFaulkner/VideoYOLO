"""Darknet as YOLO backbone network."""
# pylint: disable=arguments-differ
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['DarknetV3']


# class SelfAtt(gluon.HybridBlock):
#     def __init__(self, **kwargs):
#         super(SelfAtt, self).__init__(**kwargs)
#
#         self.d = 256
#         self.r = 1
#
#         with self.name_scope():
#             self.w_1 = nn.Dense(self.d, use_bias=False)
#             self.w_2 = nn.Dense(self.r, use_bias=False)
#
#     def hybrid_forward(self, F, x):
#         _h = x.reshape((0, 0, -1))
#         _w = self.w_1(_h)
#         _w = F.tanh(_w)
#         w = self.w_2(_w)
#         _att = w.reshape((-1, 3, self.r))  # Batch * Timestep * r
#         att = F.softmax(_att, axis=1)
#
#         return x, att


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


def _conv2d(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


class DarknetBasicBlockV3(gluon.HybridBlock):
    """Darknet Basic Block. Which is a 1x1 reduce conv followed by 3x3 conv.

    Parameters
    ----------
    channel : int
        Convolution channels for 1x1 conv.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    """
    def __init__(self, channel, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(DarknetBasicBlockV3, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(_conv2d(channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        # 3x3 conv expand
        self.body.add(_conv2d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, *args):
        residual = x
        x = self.body(x)
        return x + residual


class DarknetV3(gluon.HybridBlock):
    """Darknet v3.

    Parameters
    ----------
    layers : iterable
        Description of parameter `layers`.
    channels : iterable
        Description of parameter `channels`.
    classes : int, default is 1000
        Number of classes, which determines the dense layer output channels.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Attributes
    ----------
    features : mxnet.gluon.nn.HybridSequential
        Feature extraction layers.
    output : mxnet.gluon.nn.Dense
        A classes(1000)-way Fully-Connected Layer.

    """
    def __init__(self, layers, channels, windows, classes=1000, return_features=True,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(DarknetV3, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))
        self.windows = windows
        self.return_features = return_features
        with self.name_scope():
            self.features = nn.HybridSequential()
            # first 3x3 conv
            self.features.add(_conv2d(channels[0], 3, 1, 1,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for nlayer, channel in zip(layers, channels[1:]):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                # add downsample conv with stride=2
                self.features.add(_conv2d(channel, 3, 1, 2,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetBasicBlockV3(channel // 2,
                                                          norm_layer=BatchNorm,
                                                          norm_kwargs=None))
            # output
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        # x = TimeDistributed(self.features[:7])(x)
        x = TimeDistributed(self.features[0])(x)
        x = F.expand_dims(x, axis=1)
        x = F.reshape(x, shape=(0, self.windows[0], -1, 0, 0, 0))
        x = F.max(x, axis=1)

        if self.windows[1] == 1:
            x = F.squeeze(x, axis=1)
            if self.return_features:
                a = self.features[1:15](x)
                b = self.features[15:24](a)
                c = self.features[24:](b)
            return a, b, c

        x = TimeDistributed(self.features[1])(x)
        x = F.expand_dims(x, axis=1)
        x = F.reshape(x, shape=(0, self.windows[1], -1, 0, 0, 0))
        x = F.max(x, axis=1)

        if self.windows[2] == 1:
            x = F.squeeze(x, axis=1)
            if self.return_features:
                a = self.features[2:15](x)
                b = self.features[15:24](a)
                c = self.features[24:](b)
            return a, b, c

        x = TimeDistributed(self.features[2])(x)
        x = F.expand_dims(x, axis=1)
        x = F.reshape(x, shape=(0, self.windows[2], -1, 0, 0, 0))
        x = F.max(x, axis=1)

        if self.windows[3] == 1:
            x = F.squeeze(x, axis=1)
            if self.return_features:
                a = self.features[3:15](x)
                b = self.features[15:24](a)
                c = self.features[24:](b)
            return a, b, c

        x = TimeDistributed(self.features[3:6])(x)
        x = F.expand_dims(x, axis=1)
        x = F.reshape(x, shape=(0, self.windows[3], -1, 0, 0, 0))
        x = F.max(x, axis=1)

        if self.windows[4] == 1:
            x = F.squeeze(x, axis=1)
            if self.return_features:
                a = self.features[7:15](x)
                b = self.features[15:24](a)
                c = self.features[24:](b)
            return a, b, c

        x = TimeDistributed(self.features[7:15])(x)
        x = F.expand_dims(x, axis=1)
        x = F.reshape(x, shape=(0, self.windows[3], -1, 0, 0, 0))
        x = F.max(x, axis=1)

        if self.return_features:
            a = x
            b = self.features[15:24](a)
            c = self.features[24:](b)
            return a, b, c

        x = self.features[15:](x)
        # x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
        return x  # self.output(x)


def get_darknet(pretrained=False, ctx=mx.cpu(),
                root=os.path.join('models', 'definitions', 'darknet', 'weights'), **kwargs):
    """Get darknet by `version` and `num_layers` info.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default 'models/definitions/darknet/weights'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Darknet network.

    Examples
    --------
    >>> model = get_darknet('v3', pretrained=True)
    >>> print(model)

    """
    layers = [1, 2, 8, 8, 4]
    channels = [32, 64, 128, 256, 512, 1024]
    windows = [3, 3, 3, 1, 1]
    net = DarknetV3(layers, channels, windows, **kwargs)
    net.initialize()
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file('darknet53', tag=pretrained, root=root), ctx=ctx, ignore_extra=True)
    return net


if __name__ == '__main__':
    # just for debugging
    model = get_darknet(pretrained=True)

    model.summary(mx.nd.random_normal(shape=(2, 27, 3, 416, 416)))
