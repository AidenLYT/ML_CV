import torch
from torch.utils.data import Dataset
import os
import cv2
import os.path as osp
import numpy as np
from PIL import Image


class Proj_Dataset(Dataset):
    def __init__(self, annos, split, transform=None):
        '''
        annos: DataFrame([filename, cls])
        split: train | val | test
        '''

        self.annos = annos
        self.transform = transform
        self.split = split
        self.is_test = split == 'test'

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        tgt_row = self.annos.loc[idx]
        filename = tgt_row['filename']
        filepath = f'datasets/images/{filename}'
        if not osp.isfile(filepath):
            raise Exception(f"{filepath} does not exist")
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)  # RGB images
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img
        else:
            label = np.array(tgt_row['cls']).astype(np.int64)
            return img, label
