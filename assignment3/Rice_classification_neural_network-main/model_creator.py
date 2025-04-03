"""
In order for this code to work you need to extract the dataset from this link here:
https://drive.google.com/file/d/1eSp5f5ih17blcqjgxJQ1IKx9a7QXTqJT/view?usp=sharing

The folder name of the dataset should be called - Rice_Image_dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import os

from RiceDataset import make_image_names_csv, set_data_loaders, show_images
from NuralNetwork import NuralNetwork

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Labels map
labels_map = {
    0: "Arborio", 1: "Basmati", 2: "Ipsala", 3: "Jasmine", 4: "Karacadag"
}

# Run on gpu or cpu?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

# make a csv so the RiseDataset will know the img path
# make_image_names_csv(current_directory)


# Trian on 80% of the data & test on 20%
train_precent = 0.8
    

def train_model(batch_size, learning_rate, train_precent, epochs):

    train_loader, test_loader = set_data_loaders(current_directory, batch_size, train_precent)


    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")


    # Plot some images from the dataset
    # show_images(train_features, train_labels, labels_map)

    # Create the model
    model = NuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n---------------------")
        model.train_loop(train_loader, model, loss_fn, optimizer, device, t+1)
        accuracy = model.test_loop(test_loader, model, loss_fn, device)
    
    return model, accuracy
    

    # print("done!")

best_model = None
best_accuracy = 0

batch_size = 64
epochs = 1
learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

for lr in learning_rates:
    model, accuracy = train_model(batch_size, lr, train_precent, epochs)

    if best_model is None or best_accuracy < accuracy:
        best_accuracy = accuracy
        best_model = model


if best_model is not None:
    print(f"best_model accuracy: {best_accuracy}")

    # Saving and Loading Model Weights
    torch.save(best_model, 'model.pth')





