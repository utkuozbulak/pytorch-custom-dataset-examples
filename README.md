<p align="center"><img width="40%" src="data/pytorch-logo-dark.png" /></p>

--------------------------------------------------------------------------------

# PyTorch Custom Dataset Examples

There are some official custom dataset examples on PyTorch repo like [this](https://github.com/pytorch/tutorials/blob/master/beginner_source/data_loading_tutorial.py) but they still seemed a bit obscure to a beginner (like me) so I had to spend some time understanding what exactly I needed to have a fully customized dataset. So here we go.

The first and foremost part is creating a dataset class.

```python
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, a, b, c, d, transform=None):
        # stuff
        
    def __getitem__(self, index):
        # stuff
        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have
```

This is the skeleton that you have to fill to have a custom dataset. A dataset must contain following functions to be used by data loader afterwards. 

* **init** function where the initial logic happens like reading a csv, assigning parameters etc.
* **getitem** function where it returns a tuple of image and the label of the image. This function is called from dataloader like this:
```python
img, label = CustomDataset.__getitem__(99)
```
So, the index parameter is the **n**th image(as tensor) you are going to return.

* **len** function where it returns count of samples you have.

The first example is of having a csv file like following(without the headers, even though it really doesn't matter), that contains file name, label(class) and an extra operation indicator.

 File Name      | Label           | Extra Operation  |
| ------------- |:-------------:| :-----:|
| tr_0.png      | 5 | TRUE |
| tr_1.png      | 0      |   FALSE |
| tr_1.png      | 4      |    FALSE |

If we want to build a custom dataset that reads this csv file and images from a folder we can do something like following.

```python
class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        self.img_path = img_path  # Assign image path
        self.transform = transform  # Assign transform
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
```
In most of the examples, if not all, when a dataset is called, it is given a transform operation like this:
```python
transformations = transforms.Compose([transforms.ToTensor()])
custom_mnist_from_images =  CustomDatasetFromImages('path_to_csv', 'path_to_images', transformations)
```
transforms can contain more operations like normalize, random crop etc. The source code is [here](https://github.com/pytorch/vision/blob/master/torchvision/transforms.py). But at the end, it will probably contain transforms.ToTensor(). This operation turns the PIL images to tensors so that you can feed it to models. Thats why just before returning the tuple in **__getitem__** we do:
```python
# Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
```
Also, ToTensor can convert both grayscale and RGB images so you don't have to worry about how many channels you have for images.

**__len__** function is supposed to return the amount of images(or samples) you have, since we read the csv to pandas df at the beginning we can get the amount of samples as `len(self.data_info.index)`.

You can also make use of other columns in csv to do some operations, you just have to read the column and operate on image if there is an operation.
```python
...
 # Third column is for operation indicator
 self.operation = np.asarray(self.data_info.iloc[:, 2])
 ...
 
 ...
 if self.operation[index] == 'TRUE':
     # Do some operation
 ...
        
```

Yet another example might be reading an image from CSV where the value of each pixel is listed in a column. This just changes the parameters we take as an input and the logic in **__getitem__**. In the end, you just return images as tensors and their labels.

```python
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
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
```

I will continue updating this repo if I do some fancy stuff in the future that is different than these examples.
