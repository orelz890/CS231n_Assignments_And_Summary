import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import numpy as np
from torchvision.transforms import ToTensor, Lambda
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import sys
import subprocess


class RiceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # print("\n\n\n" + img_path + "\n\n\n")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def make_image_names_csv(current_directory):

    # Specify the Python file you want to run
    python_file = os.path.join(current_directory, "unite_images.py")

    # Get the Python interpreter executable
    python_executable = sys.executable

    # Run the Python file
    subprocess.run([python_executable, python_file])

def set_data_loaders(current_directory, batch_size, train_precent):

    folder_path = os.path.join(current_directory, "Rice_Image_Dataset")
    csv_file_path = os.path.join(folder_path, "image_names.csv")


    dataset = RiceDataset(annotations_file=csv_file_path, img_dir=folder_path, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the data
    ]))


    train_size = int(len(dataset) * train_precent)
    test_size = len(dataset) - train_size  
    
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


def show_images(train_features, train_labels, labels_map):

    # Lets see some of the images in our dataset
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_features), size=(1,)).item()
        img = train_features[sample_idx]
        label = train_labels[sample_idx].item()
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.show()
