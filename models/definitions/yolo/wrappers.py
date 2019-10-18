import platform
import warnings
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo import get_model

from .yolo3 import get_yolov3, YOLOV3_noback, YOLOV3, YOLOV3T, TimeDistributed, YOLOV3TS
from ..darknet.darknet import darknet53
from ..mobilenet.mobilenet import get_mobilenet
from ..flownet.flownet import get_flownet
from ..rdnet.r21d import get_r21d

def yolo3_darknet53(classes, dataset_name, transfer=None, pretrained_base=True, pretrained=False,
                    norm_layer=BatchNorm, norm_kwargs=None, freeze_base=False,
                    k=None, k_join_type=None, k_join_pos=None, block_conv_type='2', rnn_pos=None,
                    corr_pos=None, corr_d=None, motion_stream=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on any dataset. Modified from:
    https://github.com/dmlc/gluon-cv/blob/0dbd05c5eb8537c25b64f0e87c09be979303abf2/gluoncv/model_zoo/yolo/yolo3.py

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    dataset_name : str
        The name of the dataset, used for model save name
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from yolo networks trained on other
        datasets.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        base_net = darknet53(
            pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

        motion_net = None
        if motion_stream == 'flownet':
            assert k == 3
            motion_net = get_flownet('S', pretrained=pretrained_base, return_features=True)
        elif motion_stream == 'r21d':
            assert k in [9, 33]
            motion_net = get_r21d(34, 400, t=k-1, pretrained=pretrained_base, return_features=True)

        if freeze_base:
            for param in base_net.collect_params().values():
                param.grad_req = 'null'

            if motion_net is not None:
                for param in motion_net.collect_params().values():
                    param.grad_req = 'null'

        # if input_channels != 3:
        #     assert input_channels % 3 == 0
        #     base_net.collect_params()['darknetv30_conv0_weight']._data = \
        #         [mx.nd.repeat(base_net.collect_params()['darknetv30_conv0_weight'].data(), int(input_channels/3.0), axis=1)]
        #     base_net.collect_params()['darknetv30_conv0_weight']._shape = (32, input_channels, 3, 3)
        #     base_net.collect_params()['darknetv30_conv0_weight'].shape = (32, input_channels, 3, 3)

        stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
        strides = [8, 16, 32]
        rnn_shapes = None
        if rnn_pos is not None:
            rnn_shapes = [(1024, 13, 13), (512, 26, 26), (256, 52, 52)]  # todo currently hardcoded which will fail for input not = to 416 need to work out better way

        if motion_net is None:
            net = YOLOV3T(stages, [512, 256, 128], anchors, strides, classes=classes, k=k, k_join_type=k_join_type,
                          k_join_pos=k_join_pos, block_conv_type=block_conv_type, rnn_shapes=rnn_shapes, rnn_pos=rnn_pos,
                          corr_pos=corr_pos, corr_d=corr_d, **kwargs)
        else:
            net = YOLOV3TS(stages, motion_net, k, [512, 256, 128], anchors, strides, classes=classes, **kwargs)

    else:
        return NotImplementedError
        # net = get_model('yolo3_darknet53_' + str(transfer), pretrained=True, **kwargs)
        # reuse_classes = [x for x in classes if x in net.classes]
        # net.reset_class(classes, reuse_weights=reuse_classes)
    return net


def yolo3_no_backbone(classes, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale without the backbone network. Modified from:
    https://github.com/dmlc/gluon-cv/blob/0dbd05c5eb8537c25b64f0e87c09be979303abf2/gluoncv/model_zoo/yolo/yolo3.py

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network without the backbone.
    """
    
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    net = YOLOV3_noback([512, 256, 128], anchors, strides, classes=classes,
                        norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    
    return net


def yolo3_mobilenet1_0(classes, dataset_name, transfer=None, pretrained_base=True, pretrained=False,
                       norm_layer=BatchNorm, norm_kwargs=None, freeze_base=False,
                       k=None, k_join_type=None, k_join_pos=None, block_conv_type='2', rnn_pos=None,
                       corr_pos=None, corr_d=None, **kwargs):
    """YOLO3 multi-scale with mobilenet base network on custom dataset. Modified from:
    https://github.com/dmlc/gluon-cv/blob/0dbd05c5eb8537c25b64f0e87c09be979303abf2/gluoncv/model_zoo/yolo/yolo3.py

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    dataset_name : str
        The name of the dataset, used for model save name
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from yolo networks trained on other
        datasets.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        base_net = get_mobilenet(multiplier=1,
                                 pretrained=pretrained_base,
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                 **kwargs)

        if freeze_base:
            for param in base_net.collect_params().values():
                param.grad_req = 'null'

        stages = [base_net.features[:33],
                  base_net.features[33:69],
                  base_net.features[69:-2]]
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
        strides = [8, 16, 32]
        rnn_shapes = None
        if rnn_pos is not None:
            rnn_shapes = [(1024, 13, 13), (512, 26, 26), (256, 52, 52)]  # todo currently hardcoded which will fail for input not = to 416 need to work out better way
        # net = get_yolov3(
        #     'mobilenet1.0', stages, [512, 256, 128], anchors, strides, classes, dataset_name,
        #     norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)  # don't need get_yolov3 as won't use gluon pretrained
        # net = YOLOV3(stages, [512, 256, 128], anchors, strides, classes=classes, **kwargs)
        net = YOLOV3T(stages, [512, 256, 128], anchors, strides, classes=classes, k=k, k_join_type=k_join_type,
                      k_join_pos=k_join_pos, block_conv_type=block_conv_type, rnn_shapes=rnn_shapes, rnn_pos=rnn_pos,
                      corr_pos=corr_pos, corr_d=corr_d, **kwargs)
    else:
        net = get_model('yolo3_mobilenet1.0_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net
