import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from workspace_utils import active_session

import numpy as np

# Predict the class (or classes) of an image using a trained deep learning model.
def predict(processed_image, model, use_gpu, topk=5):
    # TODO: Implement the code to predict the class from an image file
    # im is a NumPy array - converting to tensor
    imT = torch.from_numpy(processed_image)
    # Load image to cuda if gpu is used
    if use_gpu:
        imT = imT.cuda()
    imT = imT.unsqueeze(0).float()
    # Make sure that we don't drop any nodes (similar to training and validation runs)
    model.eval()
    
    # Run the image through the model
    output = model.forward(imT)
    
    ps = torch.exp(output)
    
    # Get the top K predictions
    prob, idx = ps.topk(topk)
    
    # Define lists that will be returned by the function
    ret_prob = []
    ret_class = []
    
    # Populate the probabilties list
    for x in prob.data[0]:
        ret_prob.append(x.item()) # We have to get value out of the tensor - that's why item() is used
    
    # Populate the classes list
    # Populate the class list
    for x in idx.data[0]:
        for flower_class, dict_idx in model.class_to_idx.items():
            # Search for the predictions in the values of the model.class_to_idx dictionary 
            if x == dict_idx:
                # Get the appropriate key and append it to the returned list
                ret_class.append(flower_class)
                break
    
    return ret_prob, ret_class


# Load a model from a checkpoint
def load_checkpoint(path_to_file):
    checkpoint = torch.load(path_to_file)

    # Load the pretrained model of the specified architecture
    new_model = load_pretrained_model(checkpoint['arch'])
    # Freeze parameters (I'm not sure if that's needed)
    for param in new_model.parameters():
        param.requires_grad = False
    new_model.class_to_idx = checkpoint['model_class_to_idx']
    new_model.classifier = checkpoint['classifier']
    new_model.load_state_dict(checkpoint['state_dict'])
    
    return new_model

# Save a checpoint
def save_checkpoint(model, arch, input_size, hidden_layers, save_dir, use_gpu):
    checkpoint = {'arch': arch,
                  'gpu': use_gpu,
                  'input_size': input_size,
                  'hidden_size': hidden_layers,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'model_class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    
    # Check if save_dir exists; create it if it doesn't
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    print('Chckpoint saved!')
    return 

# Validate the network (use validation data set)
def network_accuracy(model, validloader, device):
    items_correct = 0
    items_total = 0
    
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
        
            # Get model predictions
            outputs = model(images)
        
            _, predicted = torch.max(outputs.data, 1)
            items_total += labels.size(0)
            items_correct += (predicted == labels).sum().item()

    return (100 * items_correct / items_total)


# Validation routine
def validation(model, validloader, criterion, device):
    val_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            val_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
    return val_loss, accuracy


def train_network(model, trainloader, validloader, optimizer, criterion, epochs, device):
    print_every = 20
    steps = 0
    
    # Make sure that the session stays active
    with active_session():
        for e in range(epochs):
            running_loss = 0
            model.train()
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1
                # Load to cuda if available
                inputs, labels = inputs.to(device), labels.to(device)
        
                # Make sure to zero the gradient
                optimizer.zero_grad()
        
                # Forward pass through the network
                output = model.forward(inputs)
        
                # Calculate the loss
                loss = criterion(output, labels)
        
                # Backpropagation
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                if steps % print_every == 0:
                    # Make sure that the model is in eval mode 
                    model.eval()
                    
                    # Validate the network
                    validation_loss, validation_accuracy = validation(model, validloader, criterion, device)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validloader)),
                          "Validation Accuracy: {:.3f}".format(validation_accuracy/len(validloader)))
            
                    # Make sure that the network is in training mode 
                    model.train()
                    
                    running_loss = 0
    
    return model

# This function creates a new sequential classifier 
# based on the sizes of input and hidden layers.
def create_new_classifier(input_layer, hidden_layers):
    tuple_list = []
    # Create the list for the ordered dict
    # Go through the hidden_layers list and build the structure
    counter = 1
    previous_layer = input_layer
    for layer in hidden_layers:
        tuple_list.append(('fc' + str(counter), nn.Linear(previous_layer, layer)))
        tuple_list.append(('relu' + str(counter), nn.ReLU()))
        tuple_list.append(('dropout' + str(counter), nn.Dropout()))
        previous_layer = layer
        counter += 1
    tuple_list.append(('fc4', nn.Linear(previous_layer, 102)))
    tuple_list.append(('output', nn.LogSoftmax(dim=1)))
    
    # Create a new classifier
    new_classifier = nn.Sequential(OrderedDict(tuple_list))
    
    return new_classifier

# This function returns a pretrained network of given architecture
def load_pretrained_model(model_name):
    if 'alexnet' == model_name:
        model = models.alexnet(pretrained=True)
    elif 'vgg11' == model_name:
        model = models.vgg11(pretrained=True)
    elif 'vgg11_bn' == model_name:
        model = models.vgg11_bn(pretrained=True)
    elif 'vgg13' == model_name:
        model = models.vgg13(pretrained=True)
    elif 'vgg13_bn' == model_name:
        model = models.vgg13_bn(pretrained=True)
    elif 'vgg16' == model_name:
        model = models.vgg16(pretrained=True)
    elif 'vgg16_bn' == model_name:
        model = models.vgg16_bn(pretrained=True)
    elif 'vgg19' == model_name:
        model = models.vgg19(pretrained=True)
    elif 'vgg19_bn' == model_name:
        model = models.vgg19_bn(pretrained=True)
    elif 'resnet18' == model_name:
        model = models.resnet18(pretrained=True)
    elif 'resnet34' == model_name:
        model = models.resnet34(pretrained=True)
    elif 'resnet50' == model_name:
        model = models.resnet50(pretrained=True)
    elif 'resnet101' == model_name:
        model = models.resnet101(pretrained=True)
    elif 'resnet152' == model_name:
        model = models.resnet152(pretrained=True)
    elif 'squeezenet1_0' == model_name:
        model = models.squeezenet1_0(pretrained=True)
    elif 'squeezenet1_1' == model_name:
        model = models.squeezenet1_1(pretrained=True)
    elif 'densenet121' == model_name:
        model = models.densenet121(pretrained=True)
    elif 'densenet169' == model_name:
        model = models.densenet169(pretrained=True)
    elif 'densenet161' == model_name:
        model = models.densenet161(pretrained=True)
    elif 'densenet201' == model_name:
        model = models.densenet201(pretrained=True)
    else:
        # Defaulting to inception_v3
        model = models.inception_v3(pretrained=True)

    return model


