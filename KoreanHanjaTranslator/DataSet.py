
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class KoreanHanjaCharacterDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):  # Dataset indexing
        self.img_labels = pd.read_csv("HanjaCharacters.csv")
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        path = "Images"
        if idx == 0:
            img_path = os.path.join(path, "TwoStrokes.png")
        elif idx == 1:
            img_path = os.path.join(path, "SixStrokes.png")
        elif idx == 2:
            img_path = os.path.join(path, "SevenStrokes.png")
        elif idx == 3:
            img_path = os.path.join(path, "TenStrokes.png")
        elif idx == 4:
            img_path = os.path.join(path, "ThirteenStrokes.png")
        else:
            img_path = os.path.join(path, "FourteenStrokes.png")

        # img_path = os.path.join(path, "HanjaCharacters.csv")
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        # Len of Dataset
        return len(self.img_labels)