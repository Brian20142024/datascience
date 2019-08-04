#!/usr/bin/env python
# coding: utf-8

# for utility functions like loading data and preprocessing images

# Imports here
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

import json
from collections import OrderedDict
import time
from workspace_utils import active_session
import PIL
from PIL import Image
# from skimage import io, transform

# from __future__ import print_function
# import cv2
import math
import pandas as pd


def loadData(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose(
        [transforms.RandomRotation(30),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

    class_to_idx = train_data.class_to_idx

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=True)

    return (train_loader, valid_loader, test_loader, class_to_idx)


# Label mapping
def catToName():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    print(cat_to_name)
    return cat_to_name
