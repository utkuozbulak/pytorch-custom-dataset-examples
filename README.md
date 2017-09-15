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

The first example consists of having a csv file like following(without the headers, even though it really doesn't matter), that contains file name, label(class) and an extra operation indicator. This csv file pretty much shows which image belongs to which class. 

 File Name      | Label           | Extra Operation  |
| ------------- |:-------------:| :-----:|
| tr_0.png      | 5 | TRUE |
| tr_1.png      | 0      |   FALSE |
| tr_1.png      | 4      |    FALSE |

If we want to build a custom dataset that reads this csv file and images from a location we can do something like following.

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
transformations can contain more operations like normalize, random crop etc. The source code is [here](https://github.com/pytorch/vision/blob/master/torchvision/transforms.py).
