import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


# in-repo imports
from custom_dataset_from_csv import CustomDatasetFromCsvLocation, CustomDatasetFromCsvData
from custom_dataset_from_file import CustomDatasetFromFile
from cnn_model import MnistCNNModel



if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])

    # Dataset variant 1:
    # Read image data from Csv - torch transformations are taken as argument
    custom_mnist_from_csv_data = \
        CustomDatasetFromCsvData('../data/mnist_in_csv.csv',
                                 28, 28,
                                 transformations)
    # Dataset variant 2:
    # Read image locations from Csv - transformations are defined inside dataset class
    custom_mnist_from_csv_loc =  \
        CustomDatasetFromCsvLocation('../data/mnist_labels.csv')


    # Dataset variant 3:
    # Read images from a folder, image classes are embedded in file names
    # No csv is used whatsoever
    # No torch transformations are used
    # Preprocessing operations are defined inside the dataset
    custom_mnist_from_file =  \
        CustomDatasetFromFile('../data/mnist_with_class/')

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_file,
                                                    batch_size=10,
                                                    shuffle=False)

    model = MnistCNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i, (images, labels) in enumerate(mn_dataset_loader):
        images = Variable(images)
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        break

    print('A single forward-backward pass is done!')
