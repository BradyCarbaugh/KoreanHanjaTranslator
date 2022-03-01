# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


class KoreanHanjaCharacterDataset(Dataset):
    def __init__(self, root, transforms):  # Loads data
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        # xy = np.loadtxt('./HanjaCharacters.csv', delimiter=',', dtype=np.float32)
        # self.x = torch.from_numpy(xy[:, 0:])
        # self.y = torch.from_numpy(xy[:, [0]])
        # self.n_samples = xy.shape[0]

    def __getitem__(self, index):  # Dataset indexing
        img_path = os.path.join(self.root, "Images", self.imgs[index])
        # return self.x[index], self.y[index]

    def __len__(self):
        # Len of Dataset
        return len(self.imgs)


# dataset = KoreanHanjaCharacterDataset()
# first_data = dataset[0]
# features, lables = first_data
# print(features, lables)
