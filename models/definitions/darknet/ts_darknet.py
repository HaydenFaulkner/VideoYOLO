"""
Just a helper class / model definition for the two stream models with darknet
"""

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.nn import BatchNorm
from models.definitions.darknet.darknet import get_darknet
from models.definitions.flownet.flownet import get_flownet
from models.definitions.rdnet.r21d import get_r21d


class DarknetFlownet(gluon.HybridBlock):
    def __init__(self, darknet, flownet, t=3, add_type=None, **kwargs):
        """
        A two-stream darknet with flownet with communication at 4 levels

        Args:
            darknet: the darknet model (only works with 2D version)
            flownet: the flownet model
            t: the number of timesteps
            add_type: either add or mul
            **kwargs:
        """
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

        if self.add_type in ['add', 'mul']:
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
        else:
            concat3, concat4, concat5 = self.flownet(flownet_input)
            ret_da = self.darknet.features[:15](darknet_input)
            ret_db = self.darknet.features[15:24](ret_da)
            ret_dc = self.darknet.features[24:](ret_db)

        return F.concat(ret_da, concat3),  F.concat(ret_db, concat4),  F.concat(ret_dc, concat5)


class DarknetR21D(gluon.HybridBlock):
    def __init__(self, darknet, r21d, t=9, add_type=None, **kwargs):
        """
        A two-stream darknet with R(2+1)D with communication at 4 levels

        Args:
            darknet: the darknet model (only works with 2D version)
            flownet: the flownet model
            t: the number of timesteps
            add_type: either add or mul
            **kwargs:
        """
        super(DarknetR21D, self).__init__(**kwargs)
        self.t = t
        self.add_type = add_type
        with self.name_scope():
            self.darknet = darknet
            self.r21d = r21d

    def hybrid_forward(self, F, x):
        input_arr = F.split(x, num_outputs=self.t)

        darknet_input = input_arr[int(self.t / 2)]  # b,1,c,w,h
        darknet_input = F.squeeze(darknet_input, axis=1)  # b,c,w,h

        # r21d_input = input_arr[0]  # now we also pass in the middle frame so there isn't a jump in middle
        # for i in range(1, self.t):
        #     # if i != int(self.t / 2):
        #     r21d_input = F.concat(r21d_input, input_arr[i], dim=1)

        r21d_input = x
        if self.add_type in ['add', 'mul']:
            r21d_input = F.swapaxes(r21d_input, 1, 2)  # b,t,c,w,h -> b,c,t,w,h

            r3 = self.r21d.features[:4](r21d_input)
            r7 = self.r21d.features[4:5](r3)
            r13 = self.r21d.features[5:6](r7)
            r16 = self.r21d.features[6:](r13)

            # Connection 1/4
            d = self.darknet.features[:2](darknet_input)
            if self.add_type == 'add':
                db = self.darknet.features[2].body(d + F.relu(F.max(r3, axis=2)))
            elif self.add_type == 'mul':
                db = self.darknet.features[2].body(d * F.relu(F.max(r3, axis=2)))
            else:
                db = self.darknet.features[2].body(d)  # getting body doesn't do the residual add
            d = d + db  # do our own residual

            # Connection 2/4
            d = self.darknet.features[3](d)  # do the rest of the darknet, here just the one conv 128
            if self.add_type == 'add':
                db = self.darknet.features[4].body(d + F.relu(F.max(r7, axis=2)))
            elif self.add_type == 'mul':
                db = self.darknet.features[4].body(d * F.relu(F.max(r7, axis=2)))
            else:
                db = self.darknet.features[4].body(d)  # getting body doesn't do the residual add
            d = d + db  # do our own residual

            # Connection 3/4
            d = self.darknet.features[5:7](d)  # do the rest of the darknet, here another block
            if self.add_type == 'add':
                db = self.darknet.features[7].body(d + F.relu(F.max(r13, axis=2)))
            elif self.add_type == 'mul':
                db = self.darknet.features[7].body(d * F.relu(F.max(r13, axis=2)))
            else:
                db = self.darknet.features[7].body(d)  # getting body doesn't do the residual add
            d = d + db  # do our own residual

            # Connection 4/4
            d = self.darknet.features[8:15](d)  # do the rest of the darknet, here another 7 blocks
            ret_da = d  # get first output
            d = self.darknet.features[15](d)
            if self.add_type == 'add':
                db = self.darknet.features[16].body(d + F.relu(F.max(r16, axis=2)))
            elif self.add_type == 'mul':
                db = self.darknet.features[16].body(d * F.relu(F.max(r16, axis=2)))
            else:
                db = self.darknet.features[16].body(d)  # getting body doesn't do the residual add
            d = d + db  # do our own residual

            ret_db = self.darknet.features[17:24](d)
            ret_dc = self.darknet.features[24:](ret_db)

            r7 = F.Pooling(r7, kernel=(1, 2, 2), stride=(1, 2, 2), pad=(0, 0, 0), global_pool=False, pool_type='max')  # spatial
            r7 = F.max(r7, axis=2)  # temporal - works for any number of timesteps
            r13 = F.Pooling(r13, kernel=(1, 2, 2), stride=(1, 2, 2), pad=(0, 0, 0), global_pool=False,  pool_type='max')  # spatial
            r13 = F.max(r13, axis=2)  # temporal - works for any number of timesteps
            r16 = F.Pooling(r16, kernel=(1, 2, 2), stride=(1, 2, 2), pad=(0, 0, 0), global_pool=False, pool_type='max')  # spatial
            r16 = F.max(r16, axis=2)  # temporal - works for any number of timesteps
        else:
            r7, r13, r16 = self.r21d(r21d_input)
            ret_da = self.darknet.features[:15](darknet_input)
            ret_db = self.darknet.features[15:24](ret_da)
            ret_dc = self.darknet.features[24:](ret_db)

        return F.concat(ret_da, r7), F.concat(ret_db, r13),  F.concat(ret_dc, r16)


# default configurations
def get_darknet_flownet(pretrained=False, add_type=None, t=3, **kwargs):
    assert add_type in [None, 'add', 'mul']
    darknet = get_darknet(pretrained=pretrained, **kwargs)
    flownet = get_flownet('S', pretrained=pretrained, return_features=True)
    model = DarknetFlownet(darknet=darknet, flownet=flownet, t=t, add_type=add_type)
    return model


def get_darknet_r21d(pretrained=False, add_type=None, t=9, **kwargs):
    assert add_type in [None, 'add', 'mul']
    darknet = get_darknet(pretrained=pretrained, **kwargs)
    r21d = get_r21d(34, 400, t=t-1, pretrained=pretrained, return_features=True)
    model = DarknetR21D(darknet=darknet, r21d=r21d, t=t, add_type=add_type)
    return model


if __name__ == '__main__':
    # just for debugging
    pretrained_base = True
    add_type = None  # 'add'  # 'mul'

    print('DarkNet + FlowNet')
    k = 3
    model = get_darknet_flownet(pretrained=pretrained_base, add_type=add_type, t=k, norm_layer=BatchNorm, norm_kwargs=None)
    model.summary(mx.nd.random_normal(shape=(1, k, 3, 384, 384)))

    print('DarkNet + R(2+1)D')
    k = 9
    model = get_darknet_r21d(pretrained=pretrained_base, add_type=add_type, t=k, norm_layer=BatchNorm, norm_kwargs=None)
    model.summary(mx.nd.random_normal(shape=(1, k, 3, 384, 384)))
