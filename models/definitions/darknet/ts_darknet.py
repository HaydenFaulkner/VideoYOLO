"""
Just a helper class / model definition for the two stream communication models with darknet
"""


import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from models.definitions.darknet.darknet import darknet53
from models.definitions.flownet.flownet import get_flownet
from models.definitions.rdnet.r21d import get_r21d


class DarknetFlownet(gluon.HybridBlock):
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
    def __init__(self, darknet, flownet, t=3, add_type='mul', **kwargs):
        super(DarknetFlownet, self).__init__(**kwargs)
        self.t = t
        self.add_type = add_type
        with self.name_scope():

            self.darknet = darknet
            self.flownet = flownet

    def hybrid_forward(self, F, x):
        input_arr = F.split(x, num_outputs=self.t)

        darknet_input = input_arr[int(self.t / 2)]  # b,1,c,w,h
        darknet_input = F.squeeze(darknet_input, axis=1)  # b,c,w,h

        flownet_input = input_arr[0]
        for i in range(1, self.t):
            if i != int(self.t / 2):
                flownet_input = F.concat(flownet_input, input_arr[i], dim=1)

        flownet_input = F.reshape(flownet_input, shape=(0, -3, -2))

        out_conv1 = self.flownet.conv1(flownet_input)
        out_conv2 = self.flownet.conv2(out_conv1)
        out_conv3 = self.flownet.conv3(out_conv2)
        out_conv4 = self.flownet.conv4(out_conv3)

        # Connection 1/4
        d = self.darknet.features[:2](darknet_input)
        if self.add_type == 'add':
            db = self.darknet.features[2].body(d + F.relu(out_conv1))
        elif self.add_type == 'mul':
            db = self.darknet.features[2].body(d * F.relu(out_conv1))
        else:
            db = self.darknet.features[2].body(d)  # getting body doesn't do the residual add
        d = d + db  # do our own residual

        # Connection 2/4
        d = self.darknet.features[3](d)  # do the rest of the darknet, here just the one conv 128
        if self.add_type == 'add':
            db = self.darknet.features[4].body(d + F.relu(out_conv2))
        elif self.add_type == 'mul':
            db = self.darknet.features[4].body(d * F.relu(out_conv2))
        else:
            db = self.darknet.features[4].body(d)  # getting body doesn't do the residual add
        d = d + db  # do our own residual

        # Connection 3/4
        d = self.darknet.features[5:7](d)  # do the rest of the darknet, here another block
        if self.add_type == 'add':
            db = self.darknet.features[7].body(d + F.relu(out_conv3))
        elif self.add_type == 'mul':
            db = self.darknet.features[7].body(d * F.relu(out_conv3))
        else:
            db = self.darknet.features[7].body(d)  # getting body doesn't do the residual add
        d = d + db  # do our own residual

        # Connection 4/4
        d = self.darknet.features[8:15](d)  # do the rest of the darknet, here another 7 blocks
        ret_da = d  # get first output
        d = self.darknet.features[15](d)
        if self.add_type == 'add':
            db = self.darknet.features[16].body(d + F.relu(out_conv4))
        elif self.add_type == 'mul':
            db = self.darknet.features[16].body(d * F.relu(out_conv4))
        else:
            db = self.darknet.features[16].body(d)  # getting body doesn't do the residual add
        d = d + db  # do our own residual

        ret_db = self.darknet.features[17:24](d)
        ret_dc = self.darknet.features[24:](ret_db)

        # do the rest of the flownet
        out_conv5 = self.flownet.conv5(out_conv4)
        out_conv6 = self.flownet.conv6(out_conv5)

        flow6 = self.flownet.predict_flow6(out_conv6)
        flow6_up = self.flownet.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.flownet.relu11(self.flownet.deconv5(out_conv6))

        concat5 = F.concat(out_conv5, out_deconv5, flow6_up)
        flow5 = self.flownet.predict_flow5(concat5)
        flow5_up = self.flownet.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.flownet.relu12(self.flownet.deconv4(concat5))

        concat4 = F.concat(out_conv4, out_deconv4, flow5_up)
        flow4 = self.flownet.predict_flow4(concat4)
        flow4_up = self.flownet.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.flownet.relu13(self.flownet.deconv3(concat4))

        concat3 = F.concat(out_conv3, out_deconv3, flow4_up)
        return [ret_da, concat3], [ret_db, concat4], [ret_dc, concat5]

if __name__ == '__main__':
    # just for debugging
    k = 3
    motion_stream = 'flownet'
    pretrained_base = True

    motion_net = None
    if motion_stream == 'flownet':
        assert k == 3
        flownet = get_flownet('S', pretrained=pretrained_base, return_features=True)
    elif motion_stream == 'r21d':
        assert k in [9, 33]
        motion_net = get_r21d(34, 400, t=k - 1, pretrained=pretrained_base, return_features=True)

    darknet = darknet53(pretrained=pretrained_base, norm_layer=BatchNorm, norm_kwargs=None)

    model = DarknetFlownet(darknet=darknet, flownet=flownet)

    out = model.summary(mx.nd.random_normal(shape=(1, 3, 3, 384, 384)))

    print('DONE')