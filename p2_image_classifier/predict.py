# for model prediction

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
import m_helper

if __name__ == "__main__":
    # user input /path/to/image checkpoint
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('img_pth', help='Image path, enter it')
    parser.add_argument('checkpoint', help='checkpoint, enter it')
    parser.add_argument('-a', '--architecture', required=True, help='Choose the same value as in training [vgg11, vgg19]')
    parser.add_argument('hidden_units', nargs='?', default=4096, type=int, help='Choose the same value as in training')
    args = parser.parse_args()

    m_helper.sanity_chk(args.checkpoint, args.architecture, args.hidden_units, args.img_pth)
