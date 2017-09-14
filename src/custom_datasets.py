# Author: Utku Ozbulak
# utku.ozbulak@gmail.com
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset  # For custom datasets


class CustomImageDatasetFromImages(Dataset):
    def __init__(self, csv_path, folder_path, transform=None):
        self.data_info = pd.read_csv(csv_path, header=None)
        self.img_path = folder_path
        self.transform = transform
        self.labels = np.asarray(self.data_info.iloc[:, 1])

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        single_image_name = self.data_info.iloc[index][0]
        img_as_img = Image.open(self.img_path + '/' + single_image_name)
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data_info.index)

class CustomImageDatasetFromCSV(Dataset):
    def __init__(self, csv_path, folder_path, transform=None):
        self.data_info = pd.read_csv(csv_path, header=None)
        self.img_path = folder_path
        self.transform = transform
        self.labels = np.asarray(self.data_info.iloc[:, 1])

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        single_image_name = self.data_info.iloc[index][0]
        img_as_img = Image.open(self.img_path + '/' + single_image_name)
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data_info.index)

"""
train_dataset = dsets.CIFAR10(root='../data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=2,
                                               shuffle=False)


for i, (images, labels) in enumerate(train_dataset):
    z = images
    print(z)
    print(type(images))
    print(type(labels))
    break

"""

"""
a = CustomDataset('/home/ut/git/pytorch-experiments/data/imagenet_classes.csv',
                               '/home/ut/git/pytorch-experiments/data/imagenet_images')


#custom_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=10, shuffle=True)

b, c = a.__getitem__(0)
d = a.__len__()
print(d)
print(type(b))
print(b)
print(type(c))
print(c)
"""
"""
import numpy as np
a = np.zeros(1, 3)
label = torch.from_numpy(a)
print(label)
"""

"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
tmp_df = pd.read_csv('../data/imagenet_classes.csv', header=None)
img_labels = tmp_df.ix[:,  1]
print(type(img_labels))
mlb = MultiLabelBinarizer()
b = mlb.fit_transform(img_labels)
"""
"""
train_dataset = dsets.MNIST(root='../data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

print(type(train_dataset))
print(type(a))
b,c = train_dataset.__getitem__(0)
d = a.__len__()
print(d)

print(type(b))
#print(b)
print(type(c))
print(c)
"""
"""
with open(fileName, mode='rb') as file: # b is important -> binary
    fileContent = file.read()
"""