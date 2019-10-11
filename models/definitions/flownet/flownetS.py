"""
FlowNet: Learning Optical Flow with Convolutional Networks
Philipp Fischer et al.

from https://arxiv.org/pdf/1504.06852.pdf
"""
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

class FlowNetS(HybridBlock):
    """
    FlowNet S
    """
    def __init__(self, prefix='flownetS', **kwargs):
        super(FlowNetS, self).__init__(**kwargs)
        with self.name_scope():
            self.down_1 = nn.HybridSequential(prefix=prefix+'_down_1')
            self.down_1.add(nn.AvgPool2D(pool_size=2, strides=2, padding=0, ceil_mode=True, prefix='resize_data'))
            self.down_1.add(nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3, prefix='flow_conv1'))
            self.down_1.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU1'))
            self.down_1.add(nn.Conv2D(channels=128, kernel_size=5, strides=2, padding=2, prefix='conv2'))
            self.down_1.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU2'))

            self.down_2 = nn.HybridSequential(prefix=prefix+'_down_2')
            self.down_2.add(nn.Conv2D(channels=256, kernel_size=5, strides=2, padding=2, prefix='conv3'))
            self.down_2.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU3'))
            self.down_2.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, prefix='conv3_1'))
            self.down_2.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU4'))

            self.down_3 = nn.HybridSequential(prefix=prefix+'_down_3')
            self.down_3.add(nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, prefix='conv4'))
            self.down_3.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU5'))
            self.down_3.add(nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix='conv4_1'))
            self.down_3.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU6'))

            self.down_4 = nn.HybridSequential(prefix=prefix+'_down_4')
            self.down_4.add(nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, prefix='conv5'))
            self.down_4.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU7'))
            self.down_4.add(nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix='conv5_1'))
            self.down_4.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU8'))

            self.down_5 = nn.HybridSequential(prefix=prefix+'_down_5')
            self.down_5.add(nn.Conv2D(channels=1024, kernel_size=3, strides=2, padding=1, prefix='conv6'))
            self.down_5.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU9'))
            self.down_5.add(nn.Conv2D(channels=1024, kernel_size=3, strides=1, padding=1, prefix='conv6_1'))
            self.down_5.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU10'))

            self.conv1 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='Convolution1')
            self.dec5 = nn.Conv2DTranspose(channels=512, kernel_size=4, strides=2, padding=0, prefix='deconv5')
            self.relu11 = nn.LeakyReLU(alpha=0.1, prefix='ReLU11')
            self.up6to5 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=0, prefix='upsample_flow6to5')

            self.conv2 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='Convolution2')
            self.dec4 = nn.Conv2DTranspose(channels=256, kernel_size=4, strides=2, padding=0, prefix='deconv4')
            self.relu12 = nn.LeakyReLU(alpha=0.1, prefix='ReLU12')
            self.up5to4 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=0, prefix='upsample_flow5to4')

            self.conv3 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='Convolution3')
            self.dec3 = nn.Conv2DTranspose(channels=128, kernel_size=4, strides=2, padding=0, prefix='deconv3')
            self.relu13 = nn.LeakyReLU(alpha=0.1, prefix='ReLU13')
            self.up4to3 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=0, prefix='upsample_flow4to3')

            self.conv4 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='Convolution4')
            self.dec2 = nn.Conv2DTranspose(channels=64, kernel_size=4, strides=2, padding=0, prefix='deconv2')
            self.relu14 = nn.LeakyReLU(alpha=0.1, prefix='ReLU14')
            self.up3to2 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=0, prefix='upsample_flow3to2')

            # self.pool = nn.AvgPool2D(pool_size=2, strides=2, padding=0, ceil_mode=True, prefix='resize_concat5')
            self.conv5 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='Convolution5')

    def hybrid_forward(self, F, x):

        x_1 = self.down_1(x)
        x_2 = self.down_2(x_1)
        x_3 = self.down_3(x_2)
        x_4 = self.down_4(x_3)
        x_5 = self.down_5(x_4)

        x_6f = self.conv1(x_5)

        x_6a = self.dec5(x_5)
        x_6a = x_6a[:,:,1:-1,1:-1]  # crop is there a better way to do it? crop to wh of x_4
        x_6a = self.relu11(x_6a)

        x_6b = self.up6to5(x_6f)
        x_6b = x_6b[:,:,1:-1,1:-1] # crop to wh of x_4

        x_6 = F.concat(x_4, x_6a, x_6b)

        x_7f = self.conv2(x_6)

        x_7a = self.dec4(x_6)
        x_7a = x_7a[:,:,1:-1,1:-1]  # crop to wh of x_3
        x_7a = self.relu12(x_7a)

        x_7b = self.up5to4(x_7f)
        x_7b = x_7b[:,:,1:-1,1:-1] # crop to wh of x_3

        x_7 = F.concat(x_3, x_7a, x_7b)

        x_8f = self.conv3(x_7)

        x_8a = self.dec3(x_7)
        x_8a = x_8a[:,:,1:-1,1:-1]  # crop to wh of x_2
        x_8a = self.relu13(x_8a)

        x_8b = self.up4to3(x_8f)
        x_8b = x_8b[:,:,1:-1,1:-1] # crop to wh of x_2

        x_8 = F.concat(x_2, x_8a, x_8b)

        x_9f = self.conv4(x_8)

        x_9a = self.dec2(x_8)
        x_9a = x_9a[:,:,1:-1,1:-1]  # crop to wh of x_1
        x_9a = self.relu14(x_9a)

        x_9b = self.up3to2(x_9f)
        x_9b = x_9b[:,:,1:-1,1:-1] # crop to wh of x_1

        x_9 = F.concat(x_1, x_9a, x_9b)

        x_10f = self.conv5(x_9)

        # x = self.pool(x_9)
        if autograd.is_training():
            return x_10f, x_9f, x_8f, x_7f, x_6f
        return x_10f


if __name__ == '__main__':
    # just for debugging
    import mxnet as mx
    from mxnet import gluon, autograd

    net = FlowNetS()
    net.initialize()

    # mse_loss = gluon.loss.L2Loss()
    # with autograd.record():
    out = net.summary(mx.nd.ones((1, 6, 384, 512)))
    print()