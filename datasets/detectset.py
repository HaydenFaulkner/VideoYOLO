"""
A simple dataset that takes in an image list, returns the image and the idx in the file list, used for detection vis
"""
import mxnet as mx
from mxnet.gluon.data.dataset import Dataset
import numpy as np


class DetectSet(Dataset):
    def __init__(self, file_list):
        super(DetectSet, self).__init__()
        self._file_list = file_list

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, idx):
        img_path = self._file_list[idx]
        img = mx.image.imread(img_path, 1)
        label = self._load_label(idx)
        return img, label, idx

    def sample_path(self, idx):
        return self._file_list[idx]

    def _load_label(self, idx):
        img_path = self._file_list[idx]
        label = list()
        label.append([-1, -1, -1, -1, -1])
        return np.array(label)
