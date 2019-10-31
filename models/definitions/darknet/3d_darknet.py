"""Darknet as YOLO backbone network."""
# pylint: disable=arguments-differ
from __future__ import absolute_import

import math
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['DarknetV3', 'get_darknet', 'darknet53']


def _conv2d(channel, kernel, padding, strides, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel, strides=strides, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


def _conv3d(out_channels, kernel, padding, strides, groups=1, norm_layer=BatchNorm, norm_kwargs=None):
    """A common 3dconv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='3D')
    cell.add(nn.Conv3D(out_channels, kernel_size=kernel, strides=strides, padding=padding, use_bias=False,
                       groups=groups))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


def _conv21d(out_channels, kernel, padding, strides, norm_layer=BatchNorm, norm_kwargs=None):
    """R(2+1)D from 'A Closer Look at Spatiotemporal Convolutions for Action Recognition'"""
    cell = nn.HybridSequential(prefix='R(2+1)D')
    cell.add(_conv3d(out_channels, (1, kernel[1], kernel[2]), (0, padding[1], padding[2]), (1, strides[1], strides[2])))
    cell.add(_conv3d(out_channels, (kernel[0], 1, 1), (padding[0], 0, 0), (strides[0], 1, 1), in_channels=out_channels,
                     groups=out_channels))

    return cell


class TemporalGlobalMaxPool3D(gluon.HybridBlock):
    def __init__(self, layout='NCDHW', **kwargs):
        super(TemporalGlobalMaxPool3D, self).__init__(**kwargs)
        self.pool_axis = layout.find('D')

    def hybrid_forward(self, F, x, *args):
        return F.max(x, axis=self.pool_axis)


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
    def __init__(self, channel, norm_layer=BatchNorm, norm_kwargs=None, conv_type='2D', **kwargs):
        super(DarknetBasicBlockV3, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        if conv_type == '2D':
            # 1x1 reduce
            self.body.add(_conv2d(channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            # 3x3 conv expand
            self.body.add(_conv2d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        elif conv_type == '3D':
            # 1x1x1 reduce
            self.body.add(_conv3d(channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            # 3x3 conv expand
            self.body.add(_conv3d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, *args):
        residual = x
        x = self.body(x)
        return x + residual


class Darknet3D(gluon.HybridBlock):
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
    def __init__(self, layers, channels, conv_types, classes=1000,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(Darknet3D, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))
        with self.name_scope():
            self.features = nn.HybridSequential()
            if conv_types[0] == '2D':
                # first 3x3 conv
                self.features.add(_conv2d(channels[0], 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            elif conv_types[0] == '3D':
                # first 3x3x3 conv
                self.features.add(_conv3d(channels[0], 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

            past_conv_type = conv_types[0]
            for nlayer, channel, conv_type in zip(layers, channels[1:], conv_types):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                if past_conv_type == '3D' and conv_type == '2D':
                    self.features.add(TemporalGlobalMaxPool3D())  # note this breaks the feature indices

                if conv_type == '2D':
                    # add downsample conv with stride=2
                    self.features.add(_conv2d(channel, 3, 1, 2, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                elif conv_type == '3D':
                    # add downsample conv with stride=2 in spatial dims only, this allows for min temporal input of 3+
                    self.features.add(_conv3d(channel, 3, 1, (1, 2, 2), norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetBasicBlockV3(channel // 2, conv_type=conv_type, norm_layer=BatchNorm,
                                                          norm_kwargs=None))
                past_conv_type = conv_type

            if past_conv_type == '3D':  # the whole thing was 3D
                self.features.add(TemporalGlobalMaxPool3D())

            # output
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
        return self.output(x)


# default configurations
def get_darknet(pretrained=False, ctx=mx.cpu(), root=os.path.join('models', 'definitions', 'darknet', 'weights'),
                **kwargs):
    """

    Args:
        pretrained:
        ctx:
        root:
        **kwargs:

    Returns:

    """

    layers = [1, 2, 8, 8, 4]
    channels = [32, 64, 128, 256, 512, 1024]
    conv_types = ['3D', '3D', '3D', '3D', '3D']
    base_net = Darknet3D(layers, channels, ['2D', '2D', '2D', '2D', '2D'], **kwargs)
    net = Darknet3D(layers, channels, conv_types, **kwargs)
    # net.initialize()
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        base_net.load_parameters(get_model_file('darknet53', tag=pretrained, root=root), ctx=ctx)

        base_params = base_net.collect_params()
        net_params = net.collect_params()

        for layer_name in net_params:
            if len(net_params[layer_name].shape) == 5:
                repetitions = net_params[layer_name].shape[-3]
                base_weight = base_params[layer_name.replace(net_params.prefix, base_net.prefix)].data()
                base_weight = base_weight / repetitions
                new_weight = mx.nd.repeat(mx.nd.expand_dims(base_weight, axis=-3), repetitions, axis=-3)
                net.collect_params()[layer_name]._data = new_weight
            else:
                net.collect_params()[layer_name]._data = \
                    base_params[layer_name.replace(net_params.prefix, base_net.prefix)].data()
                print(layer_name)
            print()  # works for 3d, now to implement 2+1D

    return base_net


class TestNet(gluon.HybridBlock):
    def __init__(self, in_channels=3, out_channels=16, **kwargs):
        super(TestNet, self).__init__(**kwargs)

        self.two = nn.HybridSequential()
        self.two.add(nn.Conv2D(out_channels, kernel_size=3, strides=1, padding=1,
                               use_bias=False, in_channels=in_channels))
        self.two.add(BatchNorm(epsilon=1e-5, momentum=0.9))
        self.two.add(nn.LeakyReLU(0.1))

        self.three = nn.HybridSequential()
        self.three.add(nn.Conv3D(out_channels, kernel_size=3, strides=1, padding=1,
                                 use_bias=False, in_channels=in_channels))
        self.three.add(BatchNorm(epsilon=1e-5, momentum=0.9))
        self.three.add(nn.LeakyReLU(0.1))

        self.four = nn.HybridSequential()
        self.four.add(nn.Conv3D(out_channels, kernel_size=(1, 3, 3), strides=1, padding=(0, 1, 1),
                                use_bias=False, in_channels=in_channels))
        self.four.add(BatchNorm(epsilon=1e-5, momentum=0.9))
        self.four.add(nn.LeakyReLU(0.1))
        # need groups=out_channels so don't sum over mid_channels
        self.four.add(nn.Conv3D(out_channels, kernel_size=(3, 1, 1), strides=1, padding=(1, 0, 0),
                                use_bias=False, in_channels=out_channels, groups=out_channels))
        self.four.add(BatchNorm(epsilon=1e-5, momentum=0.9))
        self.four.add(nn.LeakyReLU(0.1))

    def hybrid_forward(self, F, x):
        t = x[:, :, 0, :, :]
        return self.two(t), self.three(x), self.four(x)


if __name__ == '__main__':
    # just for debugging

    # timesteps = 3
    # in_channels = 3
    # out_channels = 16
    # test_net = TestNet()
    # test_net.initialize()
    #
    # test_params = test_net.collect_params()
    #
    # base_weight = test_params['conv0_weight'].data()
    # base_weight = base_weight / timesteps
    # new_weight = mx.nd.repeat(mx.nd.expand_dims(base_weight, axis=-3), timesteps, axis=-3)
    # test_net.collect_params()['conv1_weight']._data = [new_weight]
    #
    #
    # base_weight = test_params['conv0_weight'].data()
    # base_weight = base_weight / timesteps
    # new_weight = mx.nd.expand_dims(base_weight, axis=2)
    # test_net.collect_params()['conv2_weight']._data = [new_weight]
    # test_net.collect_params()['conv3_weight']._data = [mx.nd.ones(shape=(out_channels, 1, timesteps, 1, 1))]
    #
    # inp = mx.nd.repeat(mx.nd.random_normal(shape=(1, in_channels, 1, 4, 4)), timesteps, axis=2)
    # o1, o2, o3 = test_net.forward(inp)
    # print()

    pretrained_base = True
    darknet = get_darknet(pretrained=pretrained_base, norm_layer=BatchNorm, norm_kwargs=None)

    k = 3
    darknet.initialize()
    darknet.summary(mx.nd.random_normal(shape=(1, 3, 384, 384)))
    # darknet.summary(mx.nd.random_normal(shape=(1, k, 3, 384, 384)))

    # todo fo a forward pass on an original and 3d darknet!!
