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


# for functions and classes relating to the model
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, arch, hidden_units):
    checkpoint = torch.load(filepath)
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    base_value = 256

    h, w = image.size[:2]
    if h > w:
        new_h, new_w = base_value * h / w, base_value
    else:
        new_h, new_w = base_value, base_value * w / h

    new_h, new_w = int(new_h), int(new_w)

    image = image.resize((new_h, new_w), Image.ANTIALIAS)

    # np_image = np.array(image)
    # img = np.transpose(np_image, (1,2,0))

    transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                                    ])

    img = transform(image)

    return img


'''
# To check your work, the function below converts a PyTorch tensor and
displays it in the notebook. If your `process_image` function works,
running the output through this function should return the original
image (except for the cropped out portions).
'''


def imshow(image, ax=None, title=None):
    # if ax is None:
    # fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    np_img = np.array(image)

    image = np.transpose(np_img, (1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Image needs to be clipped between 0 and 1 or
    # it looks like noise when displayed

    image = np.clip(image, 0, 1)
    # image = transforms.ToPILImage()(image).convert('RGB')

    image = std * image + mean
    # npimg = image.numpy()
    # np_img = np.array(image)
    # img = np.transpose(np_img, (0, 1, 2))

    plt.imshow(image)
    # ax.imshow(image)

    # return ax


# ## Class Prediction

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image
        using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
#     image = cv2.imread(image_path)
    img = process_image(image)
    img_reshaped = img[None, :, :, :]

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img_reshaped)

    return torch.topk(output, topk)


def probs_classes_labels(model, probs, classes):
    # Get probabilities which is called probs3
    probs2 = np.array(probs)
    print(probs2)
    probs3 = []
    for i in range(len(probs2[0])):
        probs3.append(math.exp(probs2[0][i]))
    print("probabilities: ")
    print(probs3)

    # Get classes which is called classes3
    idx_to_classes = {v: k for k, v in model.class_to_idx.items()}
    classes2 = np.array(classes)
    print(classes2)
    classes3 = []
    for i in range(len(classes2[0])):
        classes3.append(idx_to_classes[classes2[0][i]])
    print("classes: ")
    print(classes3)

    cat_to_name = project_utils.catToName()

    # Get labels refering to classes
    labels3 = []
    for i in classes3:
        labels3.append(cat_to_name[i])
    print("labels: ")
    print(labels3)
    return(probs3, classes3, labels3)


# Sanity Checking
def sanity_chk(checkpoint, architecture, hidden_units, img_pth):
    model = load_checkpoint(checkpoint, architecture, hidden_units)
    model.eval()

    # images, labels = next(iter(test_loader))
    # imshow(images[0], ax=None, title=None)

    # img_pth = "flowers/train/1/image_06734.jpg"
    im = Image.open(img_pth)
    print(im.format, im.size, im.mode)

    # imshow(process_image(im))

    probs, classes = predict(img_pth, model, 5)
    data = probs_classes_labels(model, probs, classes)

    print('Predicted category result: {}'.format(data[2]))
    print('Predicted probability result: {}'.format(data[0]))

    # df = pd.DataFrame(list(zip(data[2], data[0])),
    #                   columns=['Category', 'Probability'])

    # df.set_index('Category', inplace=True)
    # df.sort_values(by='Probability', ascending=True).plot(
    #     kind='barh', figsize=(5, 5), color="#66c2ff")
