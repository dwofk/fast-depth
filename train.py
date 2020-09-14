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
import datetime

# Import custom Dataset
DATASET_ABS_PATH = "/workspace/mnt/repositories/bayesian-visual-odometry/scripts"
sys.path.append(DATASET_ABS_PATH)
import Datasets

# Parse command args
args = utils.parse_command()

# Load parameters from JSON
params_file = "parameters.json"
training_dir, train_val_split, depth_min, depth_max, batch_size, \
    num_workers, gpu, loss, optimizer,  num_epochs, \
        stats_frequency, save_frequency, save_dir, max_checkpoints = utils.load_training_parameters(params_file)

training_dir = utils.format_dataset_path(training_dir)

# Create dataset
dataset = Datasets.FastDepthDataset(training_dir,
                                    split='train',
                                    depthMin=depth_min,
                                    depthMax=depth_max,
                                    input_shape_model=(224, 224))

# Make training/validation split
train_val_split_lengths = utils.get_train_val_split_lengths(train_val_split, len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, train_val_split_lengths)

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

# Load the model
model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)

# Loss & Optimizer
criterion = torch.nn.L1Loss()
optimizer = optim.SGD(model.parameters(),
                      lr=optimizer["lr"],
                      momentum=optimizer["momentum"],
                      weight_decay=optimizer["weight_decay"])

# # Load model checkpoint if specified
# ret, model, optimizer, start_epoch, best_loss = utils.load_checkpoint(args.resume, model, optimizer)

# if ret:
#     experiment_dir = os.path.split(args.resume)[0]
# else:
#     experiment_dir = utils.make_dir_with_date(save_dir, "fastdepth")
start_epoch = 0
best_loss = 100000
losses = []
val_losses = []
is_best_loss = False
model.to(device)
for epoch in range(num_epochs):
    current_epoch = start_epoch + epoch + 1
    running_loss = 0.0
    model.train()
    
    for i, (input, target) in enumerate(train_loader):

        # Send data to GPU
        inputs, target = input.to(device), target.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Predict
        outputs = model(inputs)

        # Loss and backprop
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if (i + 1) % stats_frequency == 0 and i != 0:
            print('[%d, %5d] loss: %.3f' %
                  (current_epoch, i + 1, running_loss / stats_frequency))
            running_loss = 0.0

    # Validation periodically
    if (epoch + 1) % save_frequency == 0 and epoch != 0:
        with torch.no_grad():
            
            model.eval()
            for i, (input, target) in enumerate(val_loader):
                inputs, target = input.to(device), target.to(device)

                # Predict
                outputs = model(inputs)

                # Loss
                val_loss = criterion(outputs, target)
                val_losses.append(val_loss.item())
            
            # Average loss
            mean_val_loss = sum(val_losses) / len(val_losses)
            print("Validation Loss [%d]: %.3f" % (current_epoch, mean_val_loss))

            # Save best loss
            is_best_loss = mean_val_loss < best_loss
            if is_best_loss:
                best_loss = mean_val_loss

        # Save checkpoint if it's a new best
        if is_best_loss:
            print("Saving new best checkpoint")
            save_path = utils.get_save_path(current_epoch, experiment_dir)
            utils.save_model(model, optimizer, save_path, current_epoch, best_loss, max_checkpoints)

print("Finished training")

# Save the final model
save_path = utils.get_save_path(num_epochs, experiment_dir)
utils.save_model(model, optimizer, save_path, start_epoch + epoch + 1, mean_val_loss, max_checkpoints)
print("Model saved to ", os.path.abspath(save_path))
