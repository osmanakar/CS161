# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:59:36 2018

@author: Osman
"""
#IMPORT MODULES
# Numpy: matrix algebra library
import numpy as np

# PyTorch: deep learning library
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# matplotlib: visualization library
import matplotlib.pyplot as plt

#%matplotlib inline
torch.set_num_threads(2)
#%% LOAD DATASET
# Define a normalization function
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download MNIST digits dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

# Make test and train set data loaders
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)
# we have 10 classes 0 to 9
classes = list(range(10))

#%% VISUALIZE INPUT IMAGES
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
plt.figure(figsize=(16,8))
imshow(torchvision.utils.make_grid(images[:32]))
plt.axis('off')

#%% NETWORK DEFINITION

class FCNet(nn.Module):
    def __init__(self):
        """
        initialize the network, define the layers
        """
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        """
        define the forward path computation
        """
        x = x.view(-1, 1*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = FCNet()

#%%
# define the loss function and optomizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

#%% TRAIN THE NETWORK

# max interations to run, 10^100 is merely a very large value used to disable the functionality.
MAX_ITERS = 10**100

iter_cnt = 0

# loop over the dataset multiple times
for epoch in range(2):  
    for i, data in enumerate(trainloader, 0):
        iter_cnt += 1
        if iter_cnt > MAX_ITERS:
            break
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))
            running_loss = 0.0

print('Finished Training')

#%% PREDICTION EXAMPLES

# get a test data batch
dataiter = iter(testloader)
images, labels = dataiter.next()

# pick the first 7 samples
images = images[:7]
labels = labels[:7]

# print images
plt.figure(figsize=(12,6))
imshow(torchvision.utils.make_grid(images))
plt.axis('off')
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(7)))

# make predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(7)))

#%% MEASURE THE TEST ACCURACY

correct = 0 # counts the number of correct classifications
total = 0 # counts the total nunber of classifications
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# calculate the average accuracy
print('Accuracy of the network on the %d test images: %d %%' % (total,
    100 * correct / total))









