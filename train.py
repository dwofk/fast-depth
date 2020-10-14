import os
import sys
import json
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.hub
import models
import utils
import matplotlib.pyplot as plt

params_file = "parameters.json"

# Import custom Dataset
DATASET_ABS_PATH = "/workspace/data/alex/bayesian-visual-odometry/scripts"
sys.path.append(DATASET_ABS_PATH)
import Datasets

# Parse command args
args = utils.parse_command()

# Load hyperparameters from JSON
training_dir, train_val_split, depth_min, depth_max, batch_size, \
    num_workers, gpu, loss_type, optimizer,  num_epochs, \
        stats_frequency, save_frequency, save_dir, max_checkpoints = utils.load_training_parameters(params_file)

# Convert from JSON format to DataLoader format
training_dir = utils.format_dataset_path(training_dir)

# Create dataset
print("Loading the dataset...")
dataset = Datasets.FastDepthDataset(training_dir,
                                    split='train',
                                    depthMin=depth_min,
                                    depthMax=depth_max,
                                    input_shape_model=(224, 224))

# Make training/validation split
train_val_split_lengths = utils.get_train_val_split_lengths(train_val_split, len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, train_val_split_lengths)
print("Train/val split: ", train_val_split_lengths)

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         pin_memory=True)

# Configure GPU
device = torch.device("cuda:{}".format(gpu) if type(gpu) is int and torch.cuda.is_available() else "cpu")
print("Training on", device)

# Load model checkpoint if specified
model_state_dict,\
optimizer_state_dict,\
start_epoch,\
best_loss = utils.load_checkpoint(args.resume)

# Create experiment directory
if model_state_dict:
    experiment_dir = os.path.split(args.resume)[0] # Use existing folder
else:
    experiment_dir = utils.make_dir_with_date(save_dir, "fastdepth") # New folder
print("Saving results to ", experiment_dir)

# Load the model
model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
if model_state_dict:
    model.load_state_dict(model_state_dict)

# Use parallel GPUs if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # Specify which GPUs to use on DGX
num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
if torch.cuda.device_count() > 1:
  print("Let's use", num_gpus, "GPUs!")
  model = nn.DataParallel(model)

# Send model to GPU(s)
# This must be done before optimizer is created if a model state_dict is being loaded
model.to(device)

# NOT currently in use
def log_l1_loss(output, target):
    loss = torch.mean(torch.abs(torch.log(output - target)))
    return loss

# Loss & Optimizer
if loss_type == "L2":
    criterion = torch.nn.MSELoss()
    print("Using L2 Loss")
else:
    criterion = torch.nn.L1Loss()
    print("Using L1 Loss")

optimizer = optim.SGD(model.parameters(),
                      lr=optimizer["lr"],
                      momentum=optimizer["momentum"],
                      weight_decay=optimizer["weight_decay"])

if optimizer_state_dict:
    optimizer.load_state_dict(optimizer_state_dict)

# Load optimizer tensors onto GPU if necessary
utils.optimizer_to(device, optimizer)

# To catch and handle Ctrl-C interrupt
try:
    losses = []
    val_losses = []
    is_best_loss = False
    for epoch in range(num_epochs):
        current_epoch = start_epoch + epoch + 1
        running_loss = 0.0
        model.train()
        
        for i, (inputs, targets) in enumerate(train_loader):

            # Send data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Predict
            outputs = model(inputs)

            # Loss and backprop
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % stats_frequency == 0 and i != 0:
                print('[%d, %5d] loss: %.3f' %
                    (current_epoch, i + 1, running_loss / stats_frequency))
                running_loss = 0.0

        # Validation each epoch
        running_val_loss = 0.0
        with torch.no_grad():
                
            model.eval()
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Predict
                outputs = model(inputs)

                # Loss
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item()

            # Mean validation loss
            mean_val_loss = running_val_loss / len(val_loader)
            val_losses.append(mean_val_loss)
            print("Validation Loss [%d]: %.3f" % (current_epoch, mean_val_loss))

        # Save periodically
        if (epoch + 1) % save_frequency == 0:
            
                # Save best loss
                # is_best_loss = mean_val_loss < best_loss
                # if is_best_loss:
                #     best_loss = mean_val_loss

            # Save checkpoint if it's a new best
            # if is_best_loss:

            print("Saving new checkpoint")
            save_path = utils.get_save_path(current_epoch, experiment_dir)
            utils.save_model(model, optimizer, save_path, current_epoch, best_loss, max_checkpoints)

    print("Finished training")

    # Save the final model
    save_path = utils.get_save_path(num_epochs, experiment_dir)
    utils.save_model(model, optimizer, save_path, current_epoch, mean_val_loss, max_checkpoints)
    print("Model saved to ", os.path.abspath(save_path))

    # Save loss plots
    save_path = os.path.join(experiment_dir, "validation_loss.png")
    utils.save_losses_plot(save_path, num_epochs, val_losses, "Validation")

except KeyboardInterrupt:
    print("Saving model and quitting...")

    save_path = utils.get_save_path(current_epoch, experiment_dir)
    utils.save_model(model, optimizer, save_path, current_epoch, mean_val_loss, max_checkpoints)
    print("Model saved to ", os.path.abspath(save_path))

    # Save loss plots
    save_path = os.path.join(experiment_dir, "validation_loss.png")
    utils.save_losses_plot(save_path, current_epoch - 1, val_losses, "Validation")
