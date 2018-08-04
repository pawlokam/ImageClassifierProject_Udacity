import os

import json

import torch
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np


# This function Scales, crops, and normalizes a PIL image for a PyTorch model,
# returns an Numpy array
def process_image(image):    
    # Open image
    im = Image.open(image)
    
    # Resize the image
    size = 256, 256
    im.thumbnail(size)
    
    # Crop image
    crop_box = (16, 16, 240, 240)
    im = im.crop(box=crop_box)
    
    # Convert the image into NumPy array
    np_im = np.array(im)
    # Normalize the NumPy array
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_im = (np_im/255 - mean) / std
    
    # Transpopse the image (color channel is going to be the first dimension)
    transposed_im = np_im.transpose(2, 0, 1)
    
    return transposed_im

# This function creates transform according to the path to data set. Function searches for train or 'test' in the 'path' -
# if it doesn't fimd it it assumes it's a 'valid' direcotry.
def create_transform(directory):
    if "train" in directory:
        transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    elif "test" in directory:
        transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])       
    return transform


# This function loads the dataset with ImageFolder according to the transformation.
def load_dataset(directory, in_transform):
    return datasets.ImageFolder(directory, transform=in_transform)


# This function creates a dataloader for a specific data set (train, test, valid)
def define_dataloader(directory, transform):
    # Define dataloader (enable shuffling for training data set)
    if "train" in directory:
        dataloader = torch.utils.data.DataLoader(load_dataset(directory, transform), batch_size=64, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(load_dataset(directory, transform), batch_size=32)
    return dataloader

# This function creates a mapping from category label to category name.
def label_mapping(path_to_json):
    cat_to_name = {}
    if os.path.isfile(path_to_json):
        with open(path_to_json, 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name