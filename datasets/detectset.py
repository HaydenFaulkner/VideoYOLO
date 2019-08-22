"""
A simple dataset that takes in an image list, returns the image and the idx in the file list, used for detection vis
"""
import mxnet as mx
from mxnet.gluon.data.dataset import Dataset
import numpy as np


class DetectSet(Dataset):
    """
    A dataset that is simply compiled of a file list
    """
    def __init__(self, file_list):
        """
        Args:
            file_list (list): a list of full paths to the image files
        """
        super(DetectSet, self).__init__()
        self._file_list = file_list

    def __len__(self):
        """
        The length of the dataset

        Returns:
            int: the number of samples in the dataset
        """
        return len(self._file_list)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx (int): index of the sample in the dataset

        Returns:
            mxnet.NDArray: input image/volume
            numpy.ndarray: label
            int: idx (if inference=True)
        """
        img_path = self.sample_path(idx)  # get the image path for the sample
        img = mx.image.imread(img_path, 1)  # load the image using mxnet's imread()
        label = self.load_label()  # load the label
        return img, label, idx

    def sample_path(self, idx):
        """
        Get a path to an image file provided the sample index

        Args:
            idx (int): the sample index, a number from 0 to len(dataset)

        Returns:
            str: path to the image file
        """
        return self._file_list[idx]

    @staticmethod
    def load_label():
        """
        Get the label for a sample, in this case there are none so just give -1's

        Returns:
            numpy.ndarray: shape (1,5) full of -1's
        """
        return np.ones((1, 5))*-1
