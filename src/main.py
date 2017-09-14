# Author: Utku Ozbulak
# utku.ozbulak@gmail.com
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from model_cnn import MnistCNNModel

from custom_datasets import CustomDatasetFromImages, CustomDatasetFromCSV


if __name__ == "__main__":

    transformations = transforms.Compose([transforms.ToTensor()])

    custom_mnist_from_images =  \
        CustomDatasetFromImages('../data/mnist_labels.csv',
                                '../data/mnist_images',
                                transformations)
    custom_mnist_from_csv = \
        CustomDatasetFromCSV('../data/mnist_in_csv.csv',
                             28, 28,
                             transformations)

    custom_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_images,
                                                        batch_size=10,
                                                        shuffle=False)

    model = MnistCNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i, (images, labels) in enumerate(custom_dataset_loader):
        images = Variable(images)
        labels = Variable(labels)
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        outputs = model(images)
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
        break
