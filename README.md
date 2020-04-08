<p align="center"><img width="40%" src="data/pytorch-logo-dark.png" /></p>

--------------------------------------------------------------------------------

# PyTorch Custom Dataset Examples

*Update after two years:* It has been a long time since I have created this repository to guide people who are getting started with pytorch (like myself back then). However, over the course of years and various projects, the way I create my datasets changed many times. I included an additional bare bone dataset [here](#a-custom-custom-custom-dataset) to show what I am currently using. I would like to note that the reason why custom datasets are called custom is because you can shape it in anyway you desire. So, it is only natural that you (the reader) will develop your way of creating custom datasets after working on different projects. Examples presented in this project are not there as the ultimate way of creating them but instead, there to show the flexibility and the possiblity of pytorch datasets. I hope this repository is/was useful in your understanding of pytorch datasets. 

There are some official custom dataset examples on PyTorch repo like [this](https://github.com/pytorch/tutorials/blob/master/beginner_source/data_loading_tutorial.py) but they still seemed a bit obscure to a beginner (like me, back then) so I had to spend some time understanding what exactly I needed to have a fully customized dataset. To save you the trouble of going through bajillions of pages, here, I decided to write down the basics of Pytorch datasets. The topics are as follows.

# Topics
- [Custom Dataset fundamentals](#custom-dataset-fundamentals)
- [Using torchvision transforms](#using-torchvision-transforms)
- [Another way to use torchvision transforms](#another-way-to-use-torchvision-transforms)
- [Incorporating pandas (reading csv)](#incorporating-pandas)
- [Incorporating pandas with more logic in `__getitem__()`](#incorporating-pandas-with-more-logic)
- [Embedding classes into file names](#a-custom-custom-custom-dataset)
- [Torchvision transforms: to use or not to use?](#a-custom-custom-custom-dataset)
- [Using data loader with custom datasets](#using-data-loader)
- [Future updates (hopefully)](#future-updates)

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

* `__init__()` function is where the initial logic happens like reading a csv, assigning transforms, filtering data, etc.
* `__getitem__()` function returns the data and labels. This function is called from dataloader like this:
```python
img, label = MyCustomDataset.__getitem__(99)  # For 99th item
```
Here, ```MyCustomDataset``` returns two things, an image and a label but that does not mean that ```__getitem__()``` is only restricted to return those. Depending on your application you can return many things.

An important thing to note is that `__getitem__()` returns a specific type for a single data point (like a tensor, numpy array etc.), otherwise, in the data loader you will get an error like:

TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.PngImagePlugin.PngImageFile'>

* `__len__()` returns count of samples you have.

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
        data = self.transformations(data)  # (3)
        
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
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
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

## A Custom-custom-custom Dataset 
It has been a long time since I have updated this repository (huh 2 years) and during that time I have completely stopped using torchvision transforms and also csvs and whatnot (unless it is absolutely necessary). One reason I have stopped using torchivion transforms is because I have coded my own transforms but more importantly, I disliked the way transforms are often given as an argument in the dataset class when they are initialized in most of the best-practice examples, when it is not the best way of doing things. More often than not, for the training datasets I have coded over time, I had to use some form of preprocessing operation (flip, mirror, pad, add noise, saturate, crop, ...) randomly and wanted to have the freedom of choosing the degree I apply them, or not. So, my datasets often have a flow like below:

```
1- Get the location of the image/data to read 
2- Read the image/data
3- Convert to numpy
4- Do some processing on numpy array (randomly)
5- Do some processing on numpy array (randomly)
6- Do some processing on numpy array (randomly)
...
15- Convert data to tensor
return location of data, name of data, data, and label
```
You can obviously apply transforms just like I listed above too, in the end, it is a matter of taste. 

Below, I'm sharing a barebone custom dataset that I'm using for most of my experiments. 

```python
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

- A working custom dataset for Imagenet with normalizations etc. *Note:* Huh, I have been wanting to do this for a long time. I will probably do this and maybe have another dataset for CIFAR too.
- A custom dataset example for encoder-decoder networks like U-Net. *Note:* There is already a U-Net dataset my students coded for their project, have a look [here](https://github.com/ugent-korea/pytorch-unet-segmentation) if you are interested.
