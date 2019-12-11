"""Darknet as YOLO backbone network."""
# pylint: disable=arguments-differ
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

from gluoncv.model_zoo.model_store import get_model_file

__all__ = ['Darknet3D', 'get_darknet']


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
    if kernel == 3:  # we need to do a special repeat pad as the zeros effect the middle correct 2d pathway
        cell.add(Conv3DRepPad(out_channels, kernel_size=kernel, strides=strides, padding=padding, use_bias=False,
                              groups=groups))
    else:
        cell.add(nn.Conv3D(out_channels, kernel_size=kernel, strides=strides, padding=padding, use_bias=False,
                           groups=groups))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


def _conv21d(out_channels, kernel, padding, strides, norm_layer=BatchNorm, norm_kwargs=None):
    """R(2+1)D from 'A Closer Look at Spatiotemporal Convolutions for Action Recognition'"""
    cell = nn.HybridSequential(prefix='R(2+1)D')
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    cell.add(nn.Conv3D(out_channels, kernel_size=(1, kernel, kernel), strides=(1, strides[1], strides[2]),
                       padding=(0, padding, padding), use_bias=False, groups=1))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))

    if kernel == 3:  # we need to do a special repeat pad as the zeros effect the middle correct 2d pathway
        cell.add(Conv3DRepPad(out_channels, kernel_size=(kernel, 1, 1), strides=(strides[0], 1, 1),
                              padding=(1, 0, 0), use_bias=False, groups=out_channels))
    else:
        cell.add(nn.Conv3D(out_channels, kernel_size=(kernel, 1, 1), strides=(strides[0], 1, 1),
                           padding=(padding, 0, 0), use_bias=False, groups=out_channels))

    # cell.add(nn.LeakyReLU(0.1))  # this breaks the imgnet pretrain flow

    return cell


class Conv3DRepPad(gluon.HybridBlock):
    """
    A 3D Conv layer that specifically pads with repeats rather than zeros
    Used for temporal padding to keep imagenet pretrained weights information flow correct
    """
    def __init__(self, out_channels, kernel_size, strides=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
                 layout='NCDHW', activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        super(Conv3DRepPad, self).__init__(**kwargs)
        self.t_axis = layout.find('D')
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.padding = padding[0]

        self.conv = nn.Conv3D(out_channels, kernel_size, strides=strides, padding=(0, padding[1], padding[2]),
                              dilation=dilation, groups=groups, layout=layout,
                              activation=activation, use_bias=use_bias, weight_initializer=weight_initializer,
                              bias_initializer=bias_initializer, in_channels=in_channels, **kwargs)

    def hybrid_forward(self, F, x, *args):
        s = F.slice_axis(x, axis=self.t_axis, begin=0, end=1)
        f = F.slice_axis(x, axis=self.t_axis, begin=-2, end=-1)

        if self.padding > 1:
            s = F.repeat(s, repeats=self.padding, axis=self.t_axis)
            f = F.repeat(f, repeats=self.padding, axis=self.t_axis)

        x = F.concat(s, F.concat(x, f, dim=self.t_axis), dim=self.t_axis)
        x = self.conv(x)
        return x


class TemporalGlobalMaxPool3D(gluon.HybridBlock):
    """
    A global max pooling layer that only pools across the D axis
    """
    def __init__(self, layout='NCDHW', **kwargs):
        super(TemporalGlobalMaxPool3D, self).__init__(**kwargs)
        self.t_axis = layout.find('D')

    def hybrid_forward(self, F, x, *args):
        return F.max(x, axis=self.t_axis)


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
    def __init__(self, channel, norm_layer=BatchNorm, norm_kwargs=None, conv_type=2, **kwargs):
        super(DarknetBasicBlockV3, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        if conv_type == 2:
            # 1x1 reduce
            self.body.add(_conv2d(channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            # 3x3 conv expand
            self.body.add(_conv2d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        elif conv_type == 3:
            # 1x1x1 reduce
            self.body.add(_conv3d(channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            # 3x3 conv expand
            self.body.add(_conv3d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        elif conv_type == 21:
            # 1x1x1 reduce
            self.body.add(_conv3d(channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            # 3x3 conv expand
            self.body.add(_conv21d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

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
    def __init__(self, layers, channels, conv_types, classes=1000, return_features=False, funnel_time=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(Darknet3D, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))

        self.return_features = return_features

        with self.name_scope():
            self.features = nn.HybridSequential()
            if conv_types[0] == 2:  # first 3x3 conv
                self.features.add(_conv2d(channels[0], 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            elif conv_types[0] == 3:  # first 3x3x3 conv
                self.features.add(_conv3d(channels[0], 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            elif conv_types[0] == 21:  # first 3x3x1 1x3 conv
                self.features.add(_conv21d(channels[0], 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

            self.conv_swap = -1
            past_conv_type = conv_types[0]
            for i, (nlayer, channel, conv_type) in enumerate(zip(layers, channels[1:], conv_types[1:])):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                if (past_conv_type == 3 or past_conv_type == 21) and conv_type == 2:
                    self.conv_swap = i+1
                    self.features.add(TemporalGlobalMaxPool3D())  # note this breaks the feature indices

                # add downsample conv with spatial stride=2
                temp_stride = 1  # keep temproal stride = 1 for constant temporal dimension
                if funnel_time:
                    temp_stride = 2  # temproal stride = 2 for reduced temporal dimension
                if conv_type == 2:
                    self.features.add(_conv2d(channel, 3, 1, 2, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                elif conv_type == 3:  # keep temproal stride = 1 for constant temporal dimension
                    self.features.add(_conv3d(channel, 3, 1, (temp_stride, 2, 2),
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                elif conv_type == 21:
                    self.features.add(_conv21d(channel, 3, 1, (temp_stride, 2, 2),
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))

                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetBasicBlockV3(channel // 2, conv_type=conv_type, norm_layer=BatchNorm,
                                                          norm_kwargs=None))
                past_conv_type = conv_type

            if past_conv_type == 3 or past_conv_type == 21:  # the whole thing was 3D
                self.conv_swap = len(conv_types)
                self.features.add(TemporalGlobalMaxPool3D())

            # output
            if not self.return_features:
                self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        if self.return_features:
            if self.conv_swap == -1:  # 2D net
                a = self.features[:15](x)
                b = self.features[15:24](a)
                c = self.features[24:](b)
            elif self.conv_swap <= 4:  # temporal pool in first set of feats
                a = self.features[:16](x)
                b = self.features[16:25](a)
                c = self.features[25:](b)
            elif self.conv_swap == 5:  # temporal pool in second set of feats
                a = self.features[:15](x)
                b = self.features[15:25](a)
                c = self.features[25:](b)
                a = F.max(a, axis=-3)  # temporal pool the first feature which still has temporal dim
            else:  # temporal pool after feats
                a = self.features[:15](x)
                b = self.features[15:24](a)
                c = self.features[24:](b)
                a = F.max(a, axis=-3)  # temporal pool the first feature which still has temporal dim
                b = F.max(b, axis=-3)  # temporal pool the second feature which still has temporal dim
            return a, b, c

        x = self.features(x)
        x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
        return self.output(x)


# default configurations
def get_darknet(pretrained=False, ctx=mx.cpu(), root=os.path.join('models', 'definitions', 'darknet', 'weights'),
                conv_types=[2, 2, 2, 2, 2, 2], return_features=False, channels_factor=1, **kwargs):
    """
    get a 2D or 2+1D or 3D darknet model with correct transfer of imagenet pretrained weights

    Args:
        pretrained: boolean - use imagenet weights
        ctx: cpu or gpu context?
        root: the root to store / load the pretrained weights
        conv_types: a list of len=6 with either 2, 21, or 3 for the conv layer type for layers upto and not including:
                    [2, 5, 10, 27, 44, 53-1]
        **kwargs:

    Returns:
        net: the network

    """

    layers = [1, 2, 8, 8, 4]
    assert channels_factor in [1, 2, 4, 8, 16]
    if channels_factor > 1:
        pretrained=False

    channels = [int(n / channels_factor) for n in [32, 64, 128, 256, 512, 1024]]
    net = Darknet3D(layers, channels, conv_types, return_features=return_features, **kwargs)
    net.initialize()
    if pretrained:
        if 3 not in conv_types and 21 not in conv_types:
            net.load_parameters(get_model_file('darknet53', tag=pretrained, root=root), ctx=ctx,
                                ignore_extra=return_features)  # we won't have the dense layers
            return net

        # transfer weights from 2D
        base_net = Darknet3D(layers, channels, [2, 2, 2, 2, 2, 2], return_features=return_features, **kwargs)
        base_net.load_parameters(get_model_file('darknet53', tag=pretrained, root=root), ctx=ctx,
                                 ignore_extra=return_features)  # we won't have the dense layers

        base_params = base_net.collect_params()
        net_params = net.collect_params()

        if 3 in conv_types:
            assert 21 not in conv_types

            for layer_name in net_params:
                if len(net_params[layer_name].shape) == 5:
                    repetitions = net_params[layer_name].shape[-3]
                    base_layer_name = layer_name.replace(net_params.prefix, base_net.prefix)
                    base_weight = base_params[base_layer_name].data()
                    base_weight = base_weight / repetitions
                    new_weight = mx.nd.repeat(mx.nd.expand_dims(base_weight, axis=-3), repetitions, axis=-3)
                    net.collect_params()[layer_name].set_data(new_weight)
                else:
                    base_layer_name = layer_name.replace(net_params.prefix, base_net.prefix)
                    new_weight = base_params[base_layer_name].data()
                    net.collect_params()[layer_name].set_data(new_weight)
        elif 21 in conv_types:
            assert 3 not in conv_types

            conv_layer_num = -1
            for layer_name in net_params:
                if len(net_params[layer_name].shape) == 5:
                    if net_params[layer_name].shape[-3] == 3 and net_params[layer_name].shape[-1] == 1:  # the temp conv
                            repetitions = net_params[layer_name].shape[-3]
                            out_channels = net_params[layer_name].shape[0]
                            new_weight = mx.nd.ones(shape=(out_channels, 1, repetitions, 1, 1))/repetitions
                            net.collect_params()[layer_name].set_data(new_weight)
                    else:  # the spatial conv
                        conv_layer_num += 1
                        repetitions = net_params[layer_name].shape[-3]
                        base_layer_name = layer_name.replace(net_params.prefix, base_net.prefix)
                        base_layer_name = base_layer_name.split('_')[0]+'_conv'+str(conv_layer_num)+'_weight'
                        base_weight = base_params[base_layer_name].data()
                        base_weight = base_weight / repetitions
                        new_weight = mx.nd.expand_dims(base_weight, axis=-3)
                        net.collect_params()[layer_name].set_data(new_weight)
                else:
                    if 'conv' in layer_name:
                        conv_layer_num += 1
                        base_layer_name = layer_name.replace(net_params.prefix, base_net.prefix)
                        base_layer_name = base_layer_name.split('_')[0]+'_conv'+str(conv_layer_num)+'_weight'
                    else:
                        base_layer_name = layer_name.replace(net_params.prefix, base_net.prefix)

                    new_weight = base_params[base_layer_name].data()
                    net.collect_params()[layer_name].set_data(new_weight)

    net.collect_params().reset_ctx(ctx)
    return net


if __name__ == '__main__':
    # just for debugging

    darknet2D = get_darknet(pretrained=True, norm_layer=BatchNorm, norm_kwargs=None, return_features=True)
    darknet3D = get_darknet(pretrained=True, norm_layer=BatchNorm, norm_kwargs=None, conv_types=[21, 21, 21, 21, 21, 2], return_features=True, channels_factor=4, funnel_time=True)
    # darknet3D = get_darknet(pretrained=True, norm_layer=BatchNorm, norm_kwargs=None, conv_types=[3, 2, 2, 2, 2, 2], return_features=True)
    # conv_types layers => [2, 5, 10, 27, 44, 53-1]

    darknet2D.summary(mx.nd.random_normal(shape=(1, 3, 384, 384)))
    darknet3D.summary(mx.nd.random_normal(shape=(1, 3, 16, 384, 384)))

    inp = mx.nd.repeat(mx.nd.random_normal(shape=(1, 3, 1, 416, 416)), 128, axis=2)
    o2 = darknet2D.forward(inp[:, :, 1, :, :])
    o3 = darknet3D.forward(inp)

    if len(o2) > 1:  # output is the features
        print(o2[0][:, :2, :2, :2])
        if len(o3[0].shape) == 5:
            print(o3[0][:, :2, :, :2, :2])
        else:
            print(o3[0][:, :2, :2, :2])
    else:  # output is the softmax
        print(o2)
        print(o3)
