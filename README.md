<p align="center"><img width="40%" src="data/pytorch-logo-dark.png" /></p>

--------------------------------------------------------------------------------

# PyTorch Custom Dataset Examples

There are some official custom dataset examples on PyTorch repo like [this](https://github.com/pytorch/tutorials/blob/master/beginner_source/data_loading_tutorial.py) but they still seemed a bit obscure to a beginner (like me) so I had to spend some time understanding what exactly I needed to have a fully customized dataset. To save you the trouble of going through bajillion of pages, here I decided to write the basics of Pytorch datasets. The topics are as follows.

# Topics
- [Custom Dataset Fundamentals](#custom-dataset-fundamentals)
- [Using Torchvision Transforms](#using-torchvision-transforms)
- [Another Way to Use Torchvision Transforms](#another-way-to-use-torchvision-transforms)
- [Incorporating Pandas (Reading csv)](#incorporating-pandas)
- [Incorporating Pandas with More Logic in `__getitem__()`](#incorporating-pandas-with-more-logic)
- [Using Data Loader with Custom Datasets](#using-data-loader)
- [What's Next?](#future-updates)

## Custom Dataset Fundamentals
The first and foremost part is creating a dataset class.

```python
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # stuff
        
    def __getitem__(self, index):
        # stuff
        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have
```

This is the skeleton that you have to fill to have a custom dataset. A dataset must contain following functions to be used by data loader later on. 

* `__init__()` function is where the initial logic happens like reading a csv, assigning transforms etc.
* `__getitem__()` function returns the data and labels. This function is called from dataloader like this:
```python
img, label = MyCustomDataset.__getitem__(99)  # For 99th item
```
So, the index parameter is the **n**th data/image (as tensor) you are going to return.

* `__len__()` returns count of samples you have.

An important thing to note is that `__getitem__()` return a specific type for a single data point (like a tensor, numpy array etc.), otherwise, in the data loader you will get an error like:

TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.PngImagePlugin.PngImageFile'>

## Using Torchvision Transforms
In most of the examples you see `transforms = None` in the `__init__()`, this is used to apply torchvision transforms to your data/image. You can find the extensive list of the transforms [here](http://pytorch.org/docs/0.2.0/torchvision/transforms.html) and [here](https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py). The most common usage of transforms is like this:

```python
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, ..., transforms=None):
        # stuff
        ...
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        ...
        data = # Some data read from a file or image
        if self.transforms is not None:
            data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (img, label)

    def __len__(self):
        return count # of how many data(images?) you have
        
if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    custom_dataset = MyCustomDataset(..., transformations)
    
```

## Another Way to Use Torchvision Transforms

Personally, I don't like having dataset transforms outside the dataset class (see (1) above). So, instead of using transforms like the example above, you can also use it like:

```python
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # stuff
        ...
        # (2) One way to do it is define transforms individually
        # When you define the transforms it calls __init__() of the transform
        self.center_crop = transforms.CenterCrop(100)
        self.to_tensor = transforms.ToTensor()
        
        # (3) Or you can still compose them like 
        self.transformations = \
            transforms.Compose([transforms.CenterCrop(100),
                                transforms.ToTensor()])
        
    def __getitem__(self, index):
        # stuff
        ...
        data = # Some data read from a file or image
        
        # When you call the transform for the second time it calls __call__() and applies the transform 
        data = self.center_crop(data)  # (2)
        data = self.to_tensor(data)  # (2)
        
        # Or you can call the composed version
        data = self.trasnformations(data)  # (3)
        
        # Note that you only need one of the implementations, (2) or (3)
        return (img, label)

    def __len__(self):
        return count # of how many data(images?) you have
        
if __name__ == '__main__':
    # Call the dataset
    custom_dataset = MyCustomDataset(...)
    
```


## Incorporating Pandas

Let's say we want to read some data from a csv with pandas. The first example is of having a csv file like following (without the headers, even though it really doesn't matter), that contains file name, label(class) and an extra operation indicator and depending on this extra operation flag we do some operation on the image.

 File Name      | Label           | Extra Operation  |
| ------------- |:-------------:| :-----:|
| tr_0.png      | 5 | TRUE |
| tr_1.png      | 0      |   FALSE |
| tr_1.png      | 4      |    FALSE |

If we want to build a custom dataset that reads image locations form this csv file then we can do something like following.

```python
class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Check if there is an operation
        some_operation = self.operation_arr[index]
        # If there is an operation
        if some_operation:
            # Do some operation on image
            # ...
            # ...
            pass
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # Call dataset
    custom_mnist_from_images =  \
        CustomDatasetFromImages('../data/mnist_labels.csv')
```

## Incorporating Pandas with More Logic

Yet another example might be reading an image from CSV where the value of each pixel is listed in a column. (Sometimes MNIST is given this way). This just changes the logic in `__getitem__()`. In the end, you just return images as tensors and their labels. The data is divided into pixels like

Label     | pixel_1           | pixel_2  | ... |
| ------------- |:-------------:| :-----:|  :-----:| 
| 1      | 50 | 99 | ... |
| 0      | 21      |   223 | ... |
| 9      | 44      |    112 | ... |



```python
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transform

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
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)
        

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
    custom_mnist_from_csv = \
        CustomDatasetFromCSV('../data/mnist_in_csv.csv', 28, 28, transformations)
        
```

## Using Data Loader
Pytorch DataLoaders just call `__getitem__()` and wrap them up to a batch. We can technically not use Data Loaders and call `__getitem__()` one at a time and feed data to the models (even though it is super convenient to use data loader). Continuing from the example above, if we assume there is a custom dataset called *CustomDatasetFromCSV* then we can call the data loader like:

```python
...
if __name__ == "__main__":
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()])
    # Define custom dataset
    custom_mnist_from_csv = \
        CustomDatasetFromCSV('../data/mnist_in_csv.csv',
                             28, 28,
                             transformations)
    # Define data loader
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv,
                                                    batch_size=10,
                                                    shuffle=False)
    
    for images, labels in mn_dataset_loader:
        # Feed the data to the model
```

The firsts argument of the dataloader is the dataset, from there  it calls `__getitem__()` of that dataset. *batch_size* determines how many individual data points will be wrapped with a single batch. If we assume a single image tensor is of size: 1x28x28 (D:1, H:28, W:28) then, with this dataloader the returned tensor will be 10x1x28x28 (Batch-Depth-Height-Width).

A note on using **multi GPU**. The way that multi gpu is used with Pytorch data loaders is that, it tries to divide the batches evenly among all GPUs you have. So, if you use batch size that is less than amount of GPUs you have, it won't be able utilize all GPUs. 

## Future Updates

I will continue updating this repository whenever I find spare time. Below, are some of the stuff I plan to include. Please let me know if you would like to see some other specific examples.

- A working custom dataset for Imagenet with normalizations etc.
- A custom dataset example for encoder-decoder networks like U-Net.
