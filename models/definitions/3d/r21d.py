# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNetV1', 'ResNetV2',
           'BasicBlockV1', 'BasicBlockV2',
           'BottleneckV1', 'BottleneckV2',
           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
           'get_resnet']

import math
import os
import _pickle as pkl

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import base
from mxnet.gluon.nn import BatchNorm

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

def _conv3d(out_channels, kernel, strides=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
            groups=1, use_bias=False, prefix=''):
    """A common 3dconv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='3D')
    cell.add(nn.Conv3D(out_channels, kernel_size=kernel, strides=strides, padding=padding, dilation=dilation,
                       groups=groups, use_bias=use_bias, prefix=prefix))
    return cell

def _conv21d(out_channels, kernel, strides=(1, 1, 1), padding=(0, 0, 0),
             in_channels=0, mid_channels=None, norm_layer=BatchNorm, norm_kwargs=None, prefix=''):
    """R(2+1)D from 'A Closer Look at Spatiotemporal Convolutions for Action Recognition'"""
    cell = nn.HybridSequential(prefix='R(2+1)D')
    if mid_channels is None:
        mid_channels = int(math.floor((kernel[0] * kernel[1] * kernel[2] * in_channels * out_channels) /
                           (kernel[1] * kernel[2] * in_channels + kernel[0] * out_channels)))


    cell.add(_conv3d(mid_channels, (1, kernel[1], kernel[2]),
                     strides=(1, strides[1], strides[2]),
                     padding=(0, padding[1], padding[2]),
                     prefix=prefix+'middle_'))

    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, prefix=prefix+'middle_',
                        **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))

    cell.add(_conv3d(out_channels, (kernel[0], 1, 1),
                     strides=(strides[0], 1, 1),
                     padding=(padding[0], 0, 0),
                     prefix=prefix))

    return cell

# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Modified with R(2+1)D convs.
    This is used for R21DV1 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, prefix='', **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix=prefix)
        self.body.add(_conv21d(channels, kernel=[3,3,3], strides=[stride,stride,stride], padding=[1, 1, 1], in_channels=in_channels, prefix=prefix+'conv_1_'))
        self.body.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix=prefix+'conv_1_'))
        self.body.add(nn.LeakyReLU(0.1))
        self.body.add(_conv21d(channels, kernel=[3,3,3], strides=[1,1,1], padding=[1, 1, 1], in_channels=channels, prefix=prefix+'conv_2_'))
        self.body.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix=prefix+'conv_2_'))

        if downsample:
            self.downsample = nn.HybridSequential(prefix=prefix)
            self.downsample.add(_conv3d(channels, kernel=[1,1,1], strides=[stride,stride,stride], padding=[0, 0, 0], prefix=prefix+'down_'))
            self.downsample.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix=prefix+'down_'))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        act = F.Activation
        x = act(residual+x, act_type='relu')

        return x


class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Modified with R(2+1)D convs.
    This is used for R21DV1 for 50, 101, 152 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, prefix='',**kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix=prefix)
        self.body.add(_conv3d(channels//4, [1, 1, 1], strides=[stride,stride,stride], prefix=prefix+'conv_1_'))
        self.body.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix=prefix+'conv_1_'))
        self.body.add(nn.LeakyReLU(0.1))
        self.body.add(_conv21d(channels//4, [3, 3, 3], strides=[1,1,1], padding=[1,1,1], in_channels=channels//4, prefix=prefix+'conv_2_'))
        self.body.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix=prefix+'conv_2_'))
        self.body.add(nn.LeakyReLU(0.1))
        self.body.add(_conv3d(channels, [1, 1, 1], strides=[1,1,1], prefix=prefix+'conv_3_'))
        self.body.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix=prefix+'conv_3_'))
        if downsample:
            self.downsample = nn.HybridSequential(prefix=prefix)
            self.downsample.add(_conv3d(channels, [1, 1, 1], strides=[stride, stride, stride], prefix=prefix+'down_'))
            self.downsample.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix=prefix+'down_'))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        act = F.Activation
        x = act(x + residual, act_type='relu')
        return x


class BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Modified with R(2+1)D convs.
    This is used for R21DV2 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, prefix='', **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv21d(channels, kernel=[3,3,3], strides=[stride, stride, stride],
                              padding=[1, 1, 1], in_channels=in_channels, prefix=prefix+'conv_1_')
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv21d(channels, kernel=[3, 3, 3], strides=[1, 1, 1],
                              padding=[1, 1, 1], in_channels=channels, prefix=prefix + 'conv_2_')

        if downsample:
            self.downsample = _conv3d(channels, kernel=[1, 1, 1], strides=[stride, stride, stride],
                                      prefix=prefix+'down_')
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        act = F.Activation
        x = act(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = act(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class BottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Modified with R(2+1)D convs.
    This is used for R21DV2 for 50, 101, 152 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, prefix='', **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3d(channels//4, [1, 1, 1], strides=[1,1,1], prefix=prefix+'conv_1_')
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv21d(channels//4, [3, 3, 3], strides=[stride,stride,stride], padding=[1,1,1], in_channels=channels//4, prefix=prefix+'conv_2_')
        self.bn3 = nn.BatchNorm()
        self.conv3 = _conv3d(channels, [1, 1, 1], strides=[1,1,1], prefix=prefix+'conv_3_')
        if downsample:
            self.downsample = _conv3d(channels, [1, 1, 1], strides=[stride, stride, stride], prefix=prefix+'down_')
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        act = F.Activation
        x = act(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = act(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = act(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


# Nets
class R21DV1(HybridBlock):
    r"""R(2+1)D model from
    `"A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    <http://arxiv.org/pdf/1711.11248>`_ paper.
    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    t : int, default 1
        number of timesteps.
    """
    def __init__(self, block, layers, channels, classes=400, t=1, **kwargs):
        super(R21DV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(_conv21d(channels[0], [3, 7, 7], strides=[1, 2, 2], padding=[1, 3, 3], in_channels=t, mid_channels=45, prefix='init_'))
            self.features.add(BatchNorm(epsilon=1e-5, momentum=0.9, prefix='init_'))
            self.features.add(nn.LeakyReLU(0.1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i]))
            self.features.add(nn.GlobalAvgPool3D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels, prefix='block1_'))
            for i in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='block%d_'%(i+2)))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        # x = self.output(x)

        return x


class R21DV2(HybridBlock):
    r"""R(2+1)D - ResNet V2 model from
    `"A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    <http://arxiv.org/pdf/1711.11248>`_ paper.
    and
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, classes=1000, t=1, **kwargs):
        super(R21DV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))

            self.features.add(_conv21d(channels[0], [3, 7, 7], strides=[1, 2, 2], padding=[1, 3, 3],
                                       in_channels=t, mid_channels=45,prefix='init_'))

            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            # self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool3D())
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels, prefix='block1_'))
            for i in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='block%d_'%(i+2)))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        # x = self.output(x)
        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [R21DV1, R21DV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


# Constructor
def get_resnet(version, num_layers, t=1, classes=400, pretrained=False, ctx=cpu(),
               root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, classes=classes, t=t, **kwargs)
    # if pretrained:
    #     from ..model_store import get_model_file
    #     net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
    #                                        root=root), ctx=ctx)
    return net

def convert_weights(model, load_path, layers=34, save_path=None, n_classes=400):
    assert load_path[-4:] == '.pkl'  # load from .pkl from github.com/facebookresearch/VMZ/blob/master/tutorials/model_zoo.md
    assert layers in [34, 152]  # only 34 and 152 layer nets avail

    if save_path is None:
        save_path = load_path[:-4] + ".params"

    with open(load_path, 'rb') as f:
        x = pkl.load(f, encoding='latin1')

    resnet_spec = {18: [2, 2, 2, 2],
                   34: [3, 4, 6, 3],
                   50: [3, 4, 6, 3],
                   101: [3, 4, 23, 3],
                   152: [3, 8, 36, 3]}

    r = list()
    for stage, blocks in enumerate(resnet_spec[d]):
        for block in range(blocks):
            r.append(['comp_%d_' % len(r), 'r21dv10_stage%d_block%d_' % (stage + 1, block + 1)])

    r += [['conv1_', 'r21dv10_init_'],
          ['_w', '_weight'],
          ['_rm', '_running_mean'],
          ['_riv', '_running_var'],
          ['_spatbn', '_conv'],
          ['last_out_L' + str(n_classes) + '_', 'r21dv10_dense0_']]
    if layers == 34:
        r += [['shortcut_projection_3_conv_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_7_conv_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_13_conv_', 'r21dv10_stage4_block1_down_'],
              ['shortcut_projection_3_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_7_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_13_', 'r21dv10_stage4_block1_down_']]
    elif layers == 152:
        r += [['shortcut_projection_0_conv_', 'r21dv10_stage1_block1_down_'],
              ['shortcut_projection_3_conv_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_11_conv_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_47_conv_', 'r21dv10_stage4_block1_down_'],
              ['shortcut_projection_0_', 'r21dv10_stage1_block1_down_'],
              ['shortcut_projection_3_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_11_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_47_', 'r21dv10_stage4_block1_down_']]

    r += [['_conv_relu_', '_'],
          ['_dense0_beta', '_dense0_bias']
          ]

    keys = x['blobs'].keys()
    comp = dict()
    for k in keys:
        kk = k
        if kk[-2:] == '_s':
            kk = kk[:-2] + '_gamma'
        if kk[-2:] == '_b':
            kk = kk[:-2] + '_beta'
        for rep in r:
            kk = kk.replace(rep[0], rep[1])
        comp[kk] = k

    found = []
    not_found = []
    pdict = model.collect_params()
    for k in pdict.keys():
        if k in comp:
            print(k + ' :: '+str(pdict[k].data().shape) + ' <--loaded from-- '+
                  comp[k]+' :: '+str(x['blobs'][comp[k]].shape))
            found.append(k)
        else:
            not_found.append(k + ' :: ' + str(pdict[k].data().shape))

    error = False
    print("Parameters from mxnet model not matched:")
    for k in not_found:
        error = True
        print(k)
    print("Parameters from pickle model not matched:")
    for k in sorted(comp):
        if k not in found:
            error = True
            print(k + " :: " + str(x['blobs'][comp[k]].shape))

    if error:
        print("The model and save params don't align, can't load.")
        return None

    # put in the params
    param = model.collect_params()
    for k in pdict.keys():
        param._params[k]._data[0] = mx.nd.array(x['blobs'][comp[k]])

    # save out the params mx style
    model.save_parameters(save_path)

    return save_path

if __name__ == '__main__':
    # just for debugging
    import mxnet as mx
    from mxnet import gluon, autograd

    t = 2
    v = 1 # 2
    d = 152 # 101, 50, 34, 18
    n_classes = 487 #sports 1m: 487, kin: 400
    mod =  get_resnet(v, d, classes=n_classes, t=t)
    mod.initialize()

    # mse_loss = gluon.loss.L2Loss()
    # with autograd.record():
    out = mod.summary(mx.nd.ones((2, 3, t*8, 112, 112)))

    # "/home/hayden/Downloads/r2plus1d_34_clip8_ft_kinetics_from_ig65m_ f128022400.pkl"
    convert_weights(mod, load_path="/home/hayden/Downloads/r2plus1d_152_sports1m_from_scratch_f127111290.pkl",
                    layers=d, n_classes=n_classes,
                    save_path="/home/hayden/Downloads/152_sports1m_f127111290.params")


    # loss = mse_loss(out, mx.nd.ones((3, 2, 4, 1, 1)))
    # loss.backward()

