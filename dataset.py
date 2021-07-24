from os import listdir
from pathlib import Path
from tempfile import mkdtemp
from warnings import warn
import shutil
import pandas as pd
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import random
from PIL import Image
from torch.utils.data.dataset import Dataset

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
size = 430

#train_data
def train_transform(Rsize):
    return transforms.Compose([
        transforms.Resize((Rsize, Rsize), interpolation=Image.BICUBIC), #이미지 크기를 Rsize, Rsize로 Resizing함
        # transforms.RandomCrop(Rsize, padding=10),  # 이미지를 Crop한 뒤에 빈 곳을 padding함
        # transforms.RandomAffine(degrees=0, scale=(0.80, 1.2)), # translate a = width(0.1), b = height(0)
        # transforms.ColorJitter(brightness=(0.8, 1.2)),
        # transforms.RandomGrayscale(p=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

#test_data
def test_transform(Rsize):
    return transforms.Compose([
        transforms.Resize((Rsize, Rsize), interpolation=Image.BICUBIC), #이미지 크기를 Rsize, Rsize로 Resizing함
        # transforms.RandomGrayscale(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

class SMOKE(Dataset):
    def __init__(self, class_info, data_path, isTrain):
        super(SMOKE, self).__init__()
        self.classes = class_info
        self.imag_path = os.path.join(data_path, 'images')
        self.train = isTrain

        self.metadata = pd.read_csv(os.path.join(data_path, 'annot.csv'))
        self.image_filenames = []
        for _, row in self.metadata.iterrows():
            img_path = os.path.join(self.imag_path, row['filenames'])
            self.image_filenames.append([img_path, row['labels']])

        if self.train == True :
            self.transform = train_transform(Rsize=size)
        else :
            self.transform = test_transform(Rsize=size)

    def __getitem__(self, index):
        if self.train == True :
            image = self.transform(Image.open(self.image_filenames[index][0]))
            label = self.image_filenames[index][1]
            return image, label
        else :
            image = self.transform(Image.open(self.image_filenames[index][0]))
            label = self.image_filenames[index][1]
            return image, label, self.image_filenames[index][0].split('/')[-1]

    def __len__(self):
        return len(self.image_filenames)