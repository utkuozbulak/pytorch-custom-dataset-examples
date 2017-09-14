# Author: Utku Ozbulak
# utku.ozbulak@gmail.com
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from model_cnn import MnistCNNModel
from custom_datasets import CustomDatasetFromImages, CustomDatasetFromCSV


transformations = transforms.Compose([transforms.ToTensor()])

custom_mnist_dataset =  \
    CustomDatasetFromImages('../data/mnist_labels.csv',
                            '../data/mnist_images',
                            transformations)

custom_mnist_from_csv = \
    CustomDatasetFromCSV('../data/mnist_in_csv.csv', 28, 28, transformations)



custom_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_dataset,
                                                    batch_size=10,
                                                    shuffle=False)

model = MnistCNNModel()
n_iters = 1
num_epochs = 1

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
