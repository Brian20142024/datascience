#!/usr/bin/env python
# coding: utf-8

# for model training

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

import argparse
import project_utils


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic(
#     'config', "InlineBackend.figure_format = 'retina'")

def chooseModel(arch, hidden_units):
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    model

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model


def do_deep_learning(model, trainloader, epochs,
                     print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0

    # Save the checkpoint
    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys(), '\n')
    print("The state dict keys of optimizer: \n\n", optimizer.state_dict().keys())

    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'opt_state_dict': optimizer.state_dict(),
                  'epochs': 3, }

    print(checkpoint['state_dict'])

    torch.save(checkpoint, 'checkpoint.pth')


def check_accuracy_on_test(testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == "__main__":
    # user input data_dir
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('data_dir', help='ImageFolder path, enter it')
    parser.add_argument('-a', '--architecture', required=True, help='Specifies model architecture [vgg11, vgg19]')
    parser.add_argument('hidden_units', nargs='?', default=4096, type=int)
    parser.add_argument('learning_rate', nargs='?', default=0.001, type=float)
    parser.add_argument('epochs', nargs='?', default=3, type=int)
    args = parser.parse_args()

    train_loader = project_utils.loadData(args.data_dir)[0]
    valid_loader = project_utils.loadData(args.data_dir)[1]
    test_loader = project_utils.loadData(args.data_dir)[2]
    class_to_idx = project_utils.loadData(args.data_dir)[3]

    model = chooseModel(args.architecture, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    with active_session():
        do_deep_learning(model, train_loader, args.epochs, 40, criterion, optimizer, 'gpu')
        check_accuracy_on_test(valid_loader)

    # ## Testing your network
    # TODO: Do validation on the test set
    with active_session():
        print("The network's accuracy measured on the test data: ")
        check_accuracy_on_test(test_loader)
