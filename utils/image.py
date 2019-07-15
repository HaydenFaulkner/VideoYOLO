import cv2
import random


def cv_plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5, class_names=None, colors=None,
                 absolute_coordinates=True):
    """
    plot bounding boxes on an image and
    """

    if len(bboxes) < 1:
        return img

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes = [[b[0]*width, b[1]*height, b[2]*width, b[3]*height] for b in bboxes]

    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores[i] < thresh:
            continue
        if labels is not None and labels[i] < 0:
            continue

        cls_id = int(labels[i]) if labels is not None else -1
        if cls_id not in colors:
            colors[cls_id] = (int(256*random.random()), int(256*random.random()), int(256*random.random()))

        xmin, ymin, xmax, ymax = [int(x) for x in bbox]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cls_id], 2)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''

        score = '{:.2f}'.format(float(scores[i])) if scores is not None else ''

        if class_name or score:
            cv2.putText(img, '{:s} {:s}'.format(class_name, score), (xmin, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        colors[cls_id], 2)

    return img
