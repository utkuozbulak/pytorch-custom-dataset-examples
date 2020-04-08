import numpy as np
from PIL import Image
import glob

import torch
from torch.utils.data.dataset import Dataset  # For custom datasets


class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_path):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms

        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.image_list = glob.glob(folder_path+'*')
        # Calculate len
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        im_as_im = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        im_as_np = np.asarray(im_as_im)/255
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        im_as_np = np.expand_dims(im_as_np, 0)
        # Some preprocessing operations on numpy array
        # ...
        # ...
        # ...

        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(im_as_np).float()

        # Get label(class) of the image based on the file name
        class_indicator_location = single_image_path.rfind('_c')
        label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len
