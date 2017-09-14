# Author: Utku Ozbulak
# utku.ozbulak@gmail.com
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset  # For custom datasets


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        self.data_info = pd.read_csv(csv_path, header=None)
        self.img_path = img_path
        self.transform = transform
        self.labels = np.asarray(self.data_info.iloc[:, 1])  # Second column is the labels
        # Third column is for operation indicator
        self.operation = np.asarray(self.data_info.iloc[:, 2])

    def __getitem__(self, index):
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]
        # Get image name from the pandas df
        single_image_name = self.data_info.iloc[index][0]
        # Open image
        img_as_img = Image.open(self.img_path + '/' + single_image_name)
        # If there is an operation
        if self.operation[index] == 'TRUE':
            # Do some operation on image
            pass
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data_info.index)


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Create an empty numpy array to fill
        img_as_np = np.ones((28, 28), dtype='uint8')
        # Fill the numpy array with data from pandas df
        for i in range(1, self.data.shape[1]):
            row_pos = (i-1) // self.height
            col_pos = (i-1) % self.width
            img_as_np[row_pos][col_pos] = self.data.iloc[index][i]
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)
