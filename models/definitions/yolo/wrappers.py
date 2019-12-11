import warnings
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo import get_model

from .yolo3 import YOLOV3_noback, YOLOV3, YOLOV3T, YOLOV3TS, YOLOV3TB
from .yolo3_temporal import YOLOV3Temporal
from ..darknet.three_darknet import get_darknet
from ..darknet.h_darknet import get_hdarknet
from ..darknet.ts_darknet import get_darknet_flownet, get_darknet_r21d
from ..mobilenet.mobilenet import get_mobilenet

def yolo3_darknet53(classes, pretrained_base=True, norm_layer=BatchNorm, norm_kwargs=None, freeze_base=False,
                    k=None, k_join_type=None, k_join_pos=None, block_conv_type='2', rnn_pos=None,
                    corr_pos=None, corr_d=None, motion_stream=None, add_type=None, agnostic=False, new_model=False,
                    hierarchical=[1,1,1,1,1], h_join_type=None, temporal=False, **kwargs):
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

    # OLD CODE
    if new_model:
        if hierarchical[0] > 1:
            darknet_model = get_hdarknet(pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                         return_features=True, windows=hierarchical, type=h_join_type, **kwargs)
            k = 1

        else:
            darknet_model = get_darknet(pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                        return_features=True, **kwargs)

        if freeze_base:
            for param in darknet_model.collect_params().values():
                param.grad_req = 'null'
    else:
        darknet = get_darknet(pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        if freeze_base:
            for param in darknet.collect_params().values():
                param.grad_req = 'null'
        stages = [darknet.features[:15], darknet.features[15:24], darknet.features[24:]]


    ts_model = None
    if motion_stream == 'flownet':
        assert k == 3
        ts_model = get_darknet_flownet(pretrained=pretrained_base, add_type=add_type, t=k, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        if freeze_base:
            for param in ts_model.collect_params().values():
                param.grad_req = 'null'

    elif motion_stream == 'r21d':
        assert k in [9, 33]
        ts_model = get_darknet_r21d(pretrained=pretrained_base, add_type=add_type, t=k, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        if freeze_base:
            for param in ts_model.collect_params().values():
                param.grad_req = 'null'

    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    rnn_shapes = None
    if rnn_pos is not None:
        rnn_shapes = [(1024, 13, 13), (512, 26, 26), (256, 52, 52)]  # todo currently hardcoded which will fail for input not = to 416 need to work out better way

    if ts_model is None:
        if new_model:
            net = YOLOV3TB(darknet_model, [512, 256, 128], anchors, strides, classes=classes, k=k,
                           k_join_type=k_join_type,
                           k_join_pos=k_join_pos, block_conv_type=block_conv_type, rnn_shapes=rnn_shapes,
                           rnn_pos=rnn_pos,
                           corr_pos=corr_pos, corr_d=corr_d, agnostic=agnostic, **kwargs)
        elif temporal:
            net = YOLOV3Temporal(stages, [512, 256, 128], anchors, strides,
                                 classes=classes, t=k, conv=int(block_conv_type), corr_d=corr_d, **kwargs)
        else:
            # OLD CODE
            net = YOLOV3T(stages, [512, 256, 128], anchors, strides, classes=classes, k=k, k_join_type=k_join_type,
                          k_join_pos=k_join_pos, block_conv_type=block_conv_type, rnn_shapes=rnn_shapes, rnn_pos=rnn_pos,
                          corr_pos=corr_pos, corr_d=corr_d, agnostic=agnostic, **kwargs)

    else:
        net = YOLOV3TS(ts_model, k, [512, 256, 128], anchors, strides, classes=classes, agnostic=agnostic,
                       **kwargs)


    return net


def yolo3_3ddarknet(classes, pretrained_base=True, norm_layer=BatchNorm, norm_kwargs=None, freeze_base=False,
                    conv_types=[2, 2, 2, 2, 2, 2], agnostic=False, **kwargs):

    darknet_model = get_darknet(pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                return_features=True, conv_types=conv_types, **kwargs)
    if freeze_base:
        for param in darknet_model.collect_params().values():
            param.grad_req = 'null'

    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]

    net = YOLOV3TB(darknet_model, [512, 256, 128], anchors, strides, classes=classes, k=1, agnostic=agnostic, **kwargs)

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
