import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List


class VOC_Detection(Dataset):
    """
    A custom Dataset for the VOC Detection data. An index number (starting from 0) and a color is assigned to each of
    the labels of the dataset.
    """
    C = 20

    index2label = ["person",
                   "bird", "cat", "cow", "dog", "horse", "sheep",
                   "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                   "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

    label2index = {label: index for index, label in enumerate(index2label)}

    label_clrs = ["#ff0000",
                  "#2e8b57", "#808000", "#800000", "#000080", "#2f4f4f", "#ffa500",
                  "#00ff00", "#ba55d3", "#00fa9a", "#00ffff", "#0000ff", "#f08080", "#ff00ff",
                  "#1e90ff", "#ffff54", "#dda0dd", "#ff1493", "#87cefa", "#ffe4c4"]

    def __init__(self, root_dir: str, split: str = 'train',
                 transforms: Optional[Callable] = None) -> None:
        """ Initialize the VOC_Detection Dataset object.

        :param root_dir: The root directory of the dataset (this directory contains two directories 'train/' and
                         'test/'.
        :param split: The split of the dataset ('train' or 'test')
        :param transforms: The transforms that are applied to the images (x) and their corresponding targets (y).
        """

        assert split == 'train' or split == 'test'
        split_dir = os.path.join(root_dir, split)

        self.img_dir = os.path.join(split_dir, "images")
        self.annot_dir = os.path.join(split_dir, "targets")
        self.pseudonyms = [filename[:-4] for filename in os.listdir(self.annot_dir)]

        self.transforms = transforms

    def __len__(self) -> int:
        """
        Return the total number of instances of the dataset.

        :return: total instances of the dataset
        """
        return len(self.pseudonyms)

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        """
        Given an index number in range [0, dataset's length) , return the corresponding image and target of the dataset.
        If transforms is defined, the images and their targets are first transformed and then return by the function.

        :param idx: The given index number
        :return: The (x,y)-pair of the image and the target
        """
        pid = self.pseudonyms[idx]
        img_path = os.path.join(self.img_dir, f'{pid}.jpg')
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')

        img = Image.open(img_path)
        target = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)                    # Remove the header
            for row in csv_reader:
                target.append([self.label2index[row[0]]] + [int(row[i]) for i in range(1, 5)])
        target = th.Tensor(target)

        if self.transforms is not None:
            img, target = self.transforms((img, target))

        return img, target
