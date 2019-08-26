"""Given a ImageNet VID dataset, compute mAP - Modified from FGFA. """

import copy
import mxnet as mx
import numpy as np


def parse_set(dataset, iou_thr=0.5, pixel_tolerance=10):
    """
    parse ImageNet VID dataset into a list of dictionarys

    Args:
        dataset: dataset object
        iou_thr (float): the standard IoU thresh to meet (default is 0.5)
        pixel_tolerance (int): for small objects this will allow a smaller IoU to be acceptable to be TP (default is 10)

    Returns:
        list : list of dicts
    """

    res = list()
    ids = dataset.sample_ids
    for idx, (_, frame, _) in enumerate(dataset):
        w = frame[:, 2] - frame[:, 0] + 1
        h = frame[:, 3] - frame[:, 1] + 1
        thr = (w*h)/((w+pixel_tolerance)*(h+pixel_tolerance))
        thr[thr > iou_thr] = iou_thr
        res.append({'bbox': frame[:, :4],
                    'label': frame[:, 4].astype(int),
                    'thr': thr,  # this is the iou threshold that needs to be met to be GT
                    'img_ids': ids[idx]})

    return res


def vid_ap(rec, prec):
    """
    average precision calculations [precision integrated to recall]

    Args:
        rec: recall
        prec: precision

    Returns:
        float : average precision
    """

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vid_eval_motion(dataset, dt, motion_ranges, area_ranges, iou_threshold=0.5, class_map=None):
    """
    Do the evaluation

    Args:
        dataset: an ImageNetVidDetection dataset instance
        dt: the detections
        motion_ranges (list): list of lists - the motion ranges
        area_ranges (list): list of lists - the area ranges
        iou_threshold (float): the IoU threshold necessary for TP (default is 0.5)
        class_map (list): a mapping between the model prediction classes and the test set labels (default is None)

    Returns:
        list: average precision array with shapes (# motion ranges, # area ranges, # classes)
    """
    classname_map = dataset.wn_classes
    gt_img_ids = dataset.sample_ids

    recs = parse_set(dataset, iou_thr=iou_threshold, pixel_tolerance=10)

    dt = np.array(dt)
    img_ids = dt[:, 0].astype(int)
    obj_labels = dt[:, 1].astype(int)
    obj_confs = dt[:, 2].astype(float)
    obj_bboxes = dt[:, 3:].astype(float)

    # sort by img_ids
    if obj_bboxes.shape[0] > 0:
        sorted_inds = np.argsort(img_ids)
        img_ids = img_ids[sorted_inds]
        obj_labels = obj_labels[sorted_inds]
        obj_confs = obj_confs[sorted_inds]
        obj_bboxes = obj_bboxes[sorted_inds, :]

    num_imgs = max(max(gt_img_ids), max(img_ids)) + 1  # maybe len not max?
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    img_id = img_ids[0]
    # sort by confidence
    for i in range(0, len(img_ids)):
        if i == len(img_ids)-1 or img_ids[i+1] != img_id:
            conf = obj_confs[start_i:i+1]
            label = obj_labels[start_i:i+1]
            bbox = obj_bboxes[start_i:i+1, :]
            sorted_inds = np.argsort(-conf)

            obj_labels_cell[img_id] = label[sorted_inds]
            obj_confs_cell[img_id] = conf[sorted_inds]
            obj_bboxes_cell[img_id] = bbox[sorted_inds, :]
            if i < len(img_ids)-1:
                img_id = img_ids[i+1]
                start_i = i+1

    ov_all = [None] * num_imgs
    # extract objects in :param classname:
    npos = np.zeros(len(classname_map))
    if class_map is not None:
        npos = np.zeros(max(class_map)+1)

    for index, rec in enumerate(recs):
        img_id = rec['img_ids']
        gt_bboxes = rec['bbox']
        gt_labels = rec['label']

        # change the class ids for the ground truths
        if class_map is not None:
            gt_labels = np.array([class_map[int(l)] for l in gt_labels.flat])
            valid_gt = np.where(gt_labels.flat >= 0)[0]
            gt_bboxes = gt_bboxes[valid_gt, :]
            gt_labels = gt_labels.flat[valid_gt].astype(int)

        num_gt_obj = len(gt_labels)

        # calculate total gt for each class
        for x in gt_labels:
            npos[x] += 1  # class: number

        labels = obj_labels_cell[img_id]
        bboxes = obj_bboxes_cell[img_id]

        num_obj = 0 if labels is None else len(labels)
        ov_obj = [None] * num_obj
        for j in range(0, num_obj):
            bb = bboxes[j, :]
            ov_gt = np.zeros(num_gt_obj)
            for k in range(0, num_gt_obj):
                bbgt = gt_bboxes[k, :]
                bi = [np.max((bb[0], bbgt[0])), np.max((bb[1], bbgt[1])), np.min((bb[2], bbgt[2])),
                      np.min((bb[3], bbgt[3]))]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                         (bbgt[2] - bbgt[0] + 1.) * \
                         (bbgt[3] - bbgt[1] + 1.) - iw * ih
                    ov_gt[k] = iw * ih / ua
            ov_obj[j] = ov_gt
        ov_all[img_id] = ov_obj

    # get motion iou gt from dataset (makes/load a json) rather than .mat file
    motion_iou = dataset.motion_ious

    ap = np.zeros((len(motion_ranges), len(area_ranges), len(classname_map)))
    gt_precent = np.zeros((len(motion_ranges), len(area_ranges), len(classname_map)+1))

    npos_bak = copy.deepcopy(npos)

    for motion_range_id, motion_range in enumerate(motion_ranges):
        for area_range_id, area_range in enumerate(area_ranges):
            tp_cell = [None] * num_imgs
            fp_cell = [None] * num_imgs

            all_motion_iou = np.concatenate(motion_iou, axis=0)
            empty_weight = sum([(all_motion_iou[i] >= motion_range[0]) & (all_motion_iou[i] <= motion_range[1])
                                for i in range(len(all_motion_iou))]) / float(len(all_motion_iou))

            for index, rec in enumerate(recs):
                img_id = rec['img_ids']
                gt_bboxes = rec['bbox']
                gt_thr = rec['thr']
                gt_labels = rec['label']
                # change the class ids for the ground truths
                if class_map is not None:
                    gt_labels = np.array([class_map[int(l)] for l in gt_labels.flat])
                    valid_gt = np.where(gt_labels.flat >= 0)[0]
                    gt_bboxes = gt_bboxes[valid_gt, :]
                    gt_labels = gt_labels.flat[valid_gt].astype(int)

                num_gt_obj = len(gt_labels)

                gt_detected = np.zeros(num_gt_obj)

                gt_motion_iou = motion_iou[index]
                ig_gt_motion = [(gt_motion_iou[i] < motion_range[0]) | (gt_motion_iou[i] > motion_range[1])
                                for i in range(len(gt_motion_iou))]
                gt_area = [(x[3] - x[1] + 1) * (x[2] - x[0] + 1) for x in gt_bboxes]
                ig_gt_area = [(area < area_range[0]) | (area > area_range[1]) for area in gt_area]

                labels = obj_labels_cell[img_id]
                bboxes = obj_bboxes_cell[img_id]

                num_obj = 0 if labels is None else len(labels)
                tp = np.zeros(num_obj)
                fp = np.zeros(num_obj)

                for j in range(0, num_obj):
                    bb = bboxes[j, :]
                    ovmax = -1
                    kmax = -1
                    ovmax_ig = -1
                    ovmax_nig = -1
                    for k in range(0, num_gt_obj):
                        ov = ov_all[img_id][j][k]
                        if (ov >= gt_thr[k]) & (ov > ovmax) & (not gt_detected[k]) & (labels[j] == gt_labels[k]):
                            ovmax = ov
                            kmax = k
                        if ig_gt_motion[k] & (ov > ovmax_ig):
                            ovmax_ig = ov
                        if (not ig_gt_motion[k]) & (ov > ovmax_nig):
                            ovmax_nig = ov

                    if kmax >= 0:
                        gt_detected[kmax] = 1
                        if (not ig_gt_motion[kmax]) & (not ig_gt_area[kmax]):
                            tp[j] = 1.0
                    else:
                        bb_area = (bb[3] - bb[1] + 1) * (bb[2] - bb[0] + 1)
                        if (bb_area < area_range[0]) | (bb_area > area_range[1]):
                            fp[j] = 0
                            continue

                        if ovmax_nig > ovmax_ig:
                            fp[j] = 1
                        elif ovmax_ig > ovmax_nig:
                            fp[j] = 0
                        elif num_gt_obj == 0:
                            fp[j] = empty_weight
                        else:
                            fp[j] = sum([1 if ig_gt_motion[i] else 0
                                         for i in range(len(ig_gt_motion))]) / float(num_gt_obj)

                tp_cell[img_id] = tp
                fp_cell[img_id] = fp

                for k in range(0, num_gt_obj):
                    label = gt_labels[k]
                    if (ig_gt_motion[k]) | (ig_gt_area[k]):
                        npos[label] = npos[label] - 1

            ap[motion_range_id][area_range_id] = calculate_ap(tp_cell, fp_cell, gt_img_ids, obj_labels_cell,
                                                              obj_confs_cell, classname_map, npos, class_map)
            gt_precent[motion_range_id][area_range_id][len(classname_map)] = \
                sum([float(npos[i]) for i in range(len(npos))]) / sum([float(npos_bak[i])
                                                                       for i in range(len(npos_bak))])
            npos = copy.deepcopy(npos_bak)

    return ap


def boxoverlap(bb, bbgt):
    """
    Calculate box IoU between two boxes

    Args:
        bb: prediction box
        bbgt: ground truth box

    Returns:
        float: the IoU
    """
    ov = 0
    iw = np.min((bb[2], bbgt[2])) - np.max((bb[0], bbgt[0])) + 1
    ih = np.min((bb[3], bbgt[3])) - np.max((bb[1], bbgt[1])) + 1
    if iw > 0 and ih > 0:
        # compute overlap as area of intersection / area of union
        intersect = iw * ih
        ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
             (bbgt[2] - bbgt[0] + 1.) * \
             (bbgt[3] - bbgt[1] + 1.) - intersect
        ov = intersect / ua
    return ov


def calculate_ap(tp_cell, fp_cell, gt_img_ids, obj_labels_cell, obj_confs_cell, classname_map, npos, class_map=None):
    """
    Calculate the AP across the classes

    Args:
        tp_cell: todo
        fp_cell: todo
        gt_img_ids: todo
        obj_labels_cell: todo
        obj_confs_cell: todo
        classname_map: todo
        npos: todo
        class_map (list): a mapping between the model prediction classes and the test set labels (default is None)

    Returns:
        list: of length number of classes of the test set, containing the AP of each class (or -100 for no gt objs)
    """
    tp_all = np.concatenate([x for x in np.array(tp_cell)[gt_img_ids] if x is not None])
    fp_all = np.concatenate([x for x in np.array(fp_cell)[gt_img_ids] if x is not None])
    obj_labels = np.concatenate([x for x in np.array(obj_labels_cell)[gt_img_ids] if x is not None])
    confs = np.concatenate([x for x in np.array(obj_confs_cell)[gt_img_ids] if x is not None])

    sorted_inds = np.argsort(-confs)
    tp_all = tp_all[sorted_inds]
    fp_all = fp_all[sorted_inds]
    obj_labels = obj_labels[sorted_inds]

    if class_map is None:  # if a class map is specified use it, otherwise use 0->0, 1->1, etc
        class_map = list(range(len(classname_map)))

    cur_ap = np.zeros(len(classname_map))
    for c in range(len(classname_map)):
        # compute precision recall
        fp = np.cumsum(fp_all[obj_labels == class_map[c]])
        tp = np.cumsum(tp_all[obj_labels == class_map[c]])
        if npos[class_map[c]] <= 0:
            cur_ap[c] = -1
        else:
            # avoid division by zero in case first detection matches a difficult ground truth
            rec = tp / npos[class_map[c]]
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            cur_ap[c] = vid_ap(rec, prec)
    return cur_ap


class VIDDetectionMetric(mx.metric.EvalMetric):

    def __init__(self, dataset, conf_score_thresh=0.05, iou_thresh=0.5, data_shape=None, class_map=None):
        """
        Detection metric for ImageNet VID task - does standard and motion evaluation

        Args:
            dataset: the test set
            conf_score_thresh (float): predictions with confident scores smaller will be discarded (default is 0.05)
            iou_thresh (float): the IoU threshold necessary for boxes to be considered true positives (default is 0.5)
                                Note - there is also a pixel tolerance for small objects to have a lower IoU threshold.
            data_shape (tuple): if provided as (height, width) bounding boxes are rescaled when saving the predictions
            class_map (list): a mapping between the model prediction classes and the test set labels (default is None)
        """
        super(VIDDetectionMetric, self).__init__('ImgNetVIDMeanAP')
        self.dataset = dataset
        self._img_ids = sorted(dataset.sample_ids)
        self._current_id = 0
        self._results = []
        self._conf_score_thresh = conf_score_thresh
        self._iou_thresh = iou_thresh
        self._class_map = class_map  # for use when model preds are diff to eval set classes

        # hard set of the motion and area ranges
        self._motion_ranges = [[0.0, 1.0], [0.0, 0.7], [0.7, 0.9], [0.9, 1.0]]
        self._area_ranges = [[0, 1e5 * 1e5], [0, 50 * 50], [50 * 50, 150 * 150], [150 * 150, 1e5 * 1e5]]

        if isinstance(data_shape, (tuple, list)):
            assert len(data_shape) == 2, "Data shape must be (height, width)"
        elif not data_shape:
            data_shape = None
        else:
            raise ValueError("data_shape must be None or tuple of int as (height, width)")
        self._data_shape = data_shape

    def reset(self):
        self._current_id = 0
        self._results = []

    def get(self):
        """Get evaluation metrics. """

        # call real update
        try:
            if not self._results:
                self._results.append([self._img_ids[0], 0, 0, [0, 0, 0, 0]])
        except IndexError:
            # invalid model may result in empty JSON results, skip it
            return ['mAP', ], ['0.0', ]

        ap = vid_eval_motion(self.dataset, self._results, self._motion_ranges, self._area_ranges,
                             iou_threshold=self._iou_thresh, class_map=self._class_map)

        names, values = [], []
        names.append('~~~~ Summary metrics ~~~~\n')
        info_str = ''
        for motion_index, motion_range in enumerate(self._motion_ranges):
            for area_index, area_range in enumerate(self._area_ranges):
                info_str += 'motion [{0:.1f} {1:.1f}], area [{2} {3} {4} {5}]\n'.format(
                    motion_range[0], motion_range[1], np.sqrt(area_range[0]), np.sqrt(area_range[0]),
                    np.sqrt(area_range[1]), np.sqrt(area_range[1]))
                info_str += 'Mean AP@{:.1f} = {:.4f}\n' \
                            '\n'.format(self._iou_thresh, np.mean([ap[motion_index][area_index][i]
                                                                   for i in range(len(ap[motion_index][area_index]))
                                                                   if ap[motion_index][area_index][i] >= 0]))
        values.append(info_str)
        for cls_ind, cls_name in enumerate(self.dataset.classes):

            names.append(cls_name)
            values.append('{:.1f}'.format(100 * ap[0, 0, cls_ind]))

        return names, values

    # pylint: disable=arguments-differ, unused-argument
    def update(self, pred_bboxes, pred_labels, pred_scores, *args, **kwargs):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until you call self.get() to return the metrics.

        Args:
            pred_bboxes (mxnet.NDArray or numpy.ndarray): Prediction bounding boxes with shape `B, N, 4`.
                                                          Where B is the size of mini-batch, N is the number of bboxes.
            pred_labels (mxnet.NDArray or numpy.ndarray): Prediction bounding boxes labels with shape `B, N`.
            pred_scores (mxnet.NDArray or numpy.ndarray): Prediction bounding boxes scores with shape `B, N`.
            *args:
            **kwargs:
        """

        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
                return np.concatenate(out, axis=0)
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        for pred_bbox, pred_label, pred_score in zip(
                *[as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores]]):
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :].astype(np.float)
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred].astype(np.float)

            imgid = self._img_ids[self._current_id]
            self._current_id += 1
            if self._data_shape is not None:
                orig_width, orig_height = self.dataset.image_size(imgid)
                height_scale = float(orig_height) / self._data_shape[0]
                width_scale = float(orig_width) / self._data_shape[1]
            else:
                height_scale, width_scale = (1., 1.)

            # for each bbox detection in each image
            for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
                if hasattr(self.dataset, 'contiguous_id_to_json'):
                    if label not in self.dataset.contiguous_id_to_json:
                        # ignore non-exist class
                        continue
                    category_id = self.dataset.contiguous_id_to_json[label]
                else:
                    category_id = int(label)

                if score < self._conf_score_thresh:
                    continue

                # rescale bboxes
                bbox[[0, 2]] *= width_scale
                bbox[[1, 3]] *= height_scale
                # DONT convert [xmin, ymin, xmax, ymax]  to [xmin, ymin, w, h]
                # bbox[2:4] -= (bbox[:2] - 1)
                # self._results.append({'image_id': imgid,
                #                       'category_id': category_id,
                #                       'bbox': bbox[:4].tolist(),
                #                       'score': score})
                self._results.append([imgid, category_id, score] + bbox[:4].tolist())
