import os
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import data_functions
import model_functions

# Argument parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory",
                    help="Directory with training, test and validation data sets. Assuming that the structure is <data_directory>/test, <data_directory>/train, <data_directory>/valid")
    parser.add_argument("--save_dir", default="training_results",
                    help="Directory in which the checkpoint will be saved")
    parser.add_argument("--arch", default="vgg11",
                       help="Type of architecture used")
    parser.add_argument("--learning_rate", type=float, default=0.00075,
                       help="Learning rate")
    parser.add_argument("--hidden_units", nargs='+', type=int, default=[4096, 4096, 1024],
                       help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--gpu", action='store_true',
                       help="Use GPU during execution")
    args = parser.parse_args()
    return args


# Main function
def main():
    
    # Parse the arguments passed by the user
    parsed_arguments = arg_parser()
    
    # Define train, test and validation directories based on the data directory passed by the user.
    # Check if those directories exist. If not break the program.
    if(os.path.isdir(parsed_arguments.data_directory + "/test") and
       os.path.isdir(parsed_arguments.data_directory + "/train") and
       os.path.isdir(parsed_arguments.data_directory + "/valid")):
        data_dir = parsed_arguments.data_directory
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        # Create a transforms for the specific directory
        train_transform = data_functions.create_transform(train_dir)
        valid_transform = data_functions.create_transform(valid_dir)
        test_transform = data_functions.create_transform(test_dir)
        
        # Create dataloader for the data sets 
        trainloader = data_functions.define_dataloader(train_dir, train_transform)
        validloader = data_functions.define_dataloader(valid_dir, valid_transform)
        testloader  = data_functions.define_dataloader(test_dir, test_transform)
        
        # Create a mapping from category label to category name
        cat_to_name = data_functions.label_mapping('cat_to_name.json')
        
        # Load a pretrained model according to architecture given by the user
        model = model_functions.load_pretrained_model(parsed_arguments.arch)
        
        # Freeze parameters in the model 
        for param in model.parameters():
            param.requires_grad = False
        
        # Define a new classifier
        new_classifier = model_functions.create_new_classifier(model.classifier[0].in_features, parsed_arguments.hidden_units)
        
        # Replace the model classifier
        model.classifier = new_classifier
        
        # Define criterion and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.00075)
        
        # Use proper device (cpu/gpu)
        device = torch.device("cuda:0" if (torch.cuda.is_available() and parsed_arguments.gpu) else "cpu")
        print(device)
        model.to(device)
        
        # Train the network
        model = model_functions.train_network(model, trainloader, validloader, optimizer, criterion, parsed_arguments.epochs, device)
        
        # Validate the network
        #validation_accuracy = model_functions.network_accuracy(model, validloader, device)
        #print('Accuracy on the validation images: %d %%' % validation_accuracy)
        
        # Check accuracy on the test dataset
        training_accuracy = model_functions.network_accuracy(model, testloader, device)
        print('Accuracy on the test images: %d %%' % training_accuracy)
        
        # Save the checkpoint
        model.class_to_idx = datasets.ImageFolder(train_dir, transform=train_transform).class_to_idx
        model_functions.save_checkpoint(model, parsed_arguments.arch, model.classifier[0].in_features, parsed_arguments.hidden_units, parsed_arguments.save_dir, parsed_arguments.gpu)
        
    else:
        print("Could not find test, train or valid folder inside of %s", parsed_arguments.data_directory)
    
    return

if __name__ == "__main__":
    main()