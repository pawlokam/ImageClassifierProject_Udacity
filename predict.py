import os
import argparse

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from workspace_utils import active_session
from PIL import Image
import numpy as np

import data_functions
import model_functions

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_image",
                    help="Path to the image that presents a flower which name is going to be predicted")
    parser.add_argument("checkpoint",
                    help="Path to the saved checkpoint")
    parser.add_argument("--category_names", default="cat_to_name.json",
                       help="File with category to name mapping")
    parser.add_argument("--top_K", type=int, default=3,
                       help="Number of most likely predictions that are going to be displayed")
    parser.add_argument("--gpu", action='store_true',
                       help="Use GPU during execution")
    args = parser.parse_args() 
    return args

def main():
    
    # Parse the arguments passed by the user
    parsed_arguments = arg_parser()
    
    # Check if both checkpoint and image file exist
    if os.path.isfile(parsed_arguments.path_to_image) and os.path.isfile(parsed_arguments.checkpoint):
        
        # Make sure that the model is available on the used device
        device = torch.device("cuda:0" if (torch.cuda.is_available() and parsed_arguments.gpu) else "cpu")
        print(device)
    
        # Load a model from a checkpoint
        loaded_model = model_functions.load_checkpoint(parsed_arguments.checkpoint)
        loaded_model.to(device)
        #print(loaded_model)
        processed_image = data_functions.process_image(parsed_arguments.path_to_image)
        
        # Don't allow the top_K to be lower than 1
        topk = parsed_arguments.top_K
        if topk < 1:
            topk = 1
        probs, classes = model_functions.predict(processed_image, loaded_model, parsed_arguments.gpu, topk)
        
        if os.path.isfile(parsed_arguments.category_names):
            # Create a mapping from category label to category name
            cat_to_name = data_functions.label_mapping(parsed_arguments.category_names)
            for item, prob in zip(classes, probs):
                print("The probability that image {} is a '{}' is {}%".format(parsed_arguments.path_to_image, cat_to_name[str(item)], prob))
        else:
            print('Failed to get category to name mapping! File does not exist: %s', parsed_argument.category_names)
            print('Probabilities {}'.format(probs))
            print('Classes {}'.format(classes))
    else:
        print("Either image or the checkpoint file doesn't exist. Please check the parameters.")
    return

if __name__ == "__main__":
    main()