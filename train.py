import os
import sys
import json
import numpy as np
from comet_ml import Experiment
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.hub
import models
import utils
import matplotlib.pyplot as plt
from metrics import AverageMeter, Result

params_file = "parameters.json"

# Import custom Dataset
#DATASET_ABS_PATH = "/workspace/mnt/repositories/bayesian-visual-odometry/scripts"
DATASET_ABS_PATH = "/workspace/data/alex/bayesian-visual-odometry/scripts"
sys.path.append(DATASET_ABS_PATH)
import Datasets

# Parse command args
args = utils.parse_command()

# Load hyperparameters from JSON
training_dir, test_dir, train_val_split, depth_min, depth_max, batch_size, \
    num_workers, gpu, loss_type, optimizer,  num_epochs, \
        stats_frequency, save_frequency, save_dir, max_checkpoints = utils.load_training_parameters(params_file)

hyper_params = {
    "learning_rate" : optimizer["lr"],
    "momentum" : optimizer["momentum"],
    "weight_decay" : optimizer["weight_decay"],
    "optimizer" : optimizer["type"],
    "loss" : loss_type,
    "num_epochs" : num_epochs,
    "batch_size" : batch_size,
    "train_val_split" : train_val_split[0],
    "depth_max" : depth_max
}

experiment = Experiment(api_key = "Bq3mQixNCv2jVzq2YBhLdxq9A", project_name="fastdepth")
experiment.log_parameters(hyper_params)
experiment.add_tag(str(loss_type))

# Convert from JSON format to DataLoader format
training_dir = utils.format_dataset_path(training_dir)
test_dir = utils.format_dataset_path(test_dir)

# Create dataset
print("Loading the dataset...")
dataset = Datasets.FastDepthDataset(training_dir,
                                    split='train',
                                    depthMin=depth_min,
                                    depthMax=depth_max,
                                    input_shape_model=(224, 224))

test_dataset = Datasets.FastDepthDataset(test_dir,
                                    split='val',
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

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=True)

# Configure GPU
device = torch.device("cuda:{}".format(gpu) if type(gpu) is int and torch.cuda.is_available() else "cpu")

# Load model checkpoint if specified
model_state_dict,\
optimizer_state_dict,\
start_epoch,\
best_loss = utils.load_checkpoint(args.resume)
model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)

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
utils.optimizer_to_gpu(optimizer)

# To catch and handle Ctrl-C interrupt
try:
    with experiment.train():
        train_step = 0
        val_step = 0
        for epoch in range(num_epochs):
            experiment.log_current_epoch(epoch)
            current_epoch = start_epoch + epoch + 1
            running_loss = 0.0
            model.train()
            img_idxs = np.random.randint(0, len(train_loader), size=5)
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

                result = Result()
                result.evaluate(outputs.data, targets.data)

                # Log to Comet
                experiment.log_metric("loss", loss.item(), step=train_step)
                experiment.log_metric("rmse", result.rmse, step=train_step)
                experiment.log_metric("mae", result.mae, step=train_step)
                experiment.log_metric("delta1", result.delta1, step=train_step)
                train_step += 1

                # Save some images for Comet
                if i in img_idxs:
                    utils.log_image_to_comet(inputs[0], targets[0], outputs[0], epoch, i, experiment, "train")

                # Print statistics
                running_loss += loss.item()
                if (i + 1) % stats_frequency == 0 and i != 0:
                    print('[%d, %5d] loss: %.3f' %
                        (current_epoch, i + 1, running_loss / stats_frequency))
                    running_loss = 0.0

            # Validation each epoch
            running_val_loss = 0.0
            with torch.no_grad():
                img_idxs = np.random.randint(0, len(val_loader), size=5)
                model.eval()
                for i, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Predict
                    outputs = model(inputs)

                    # Loss
                    val_loss = criterion(outputs, targets)

                    running_val_loss += val_loss.item()
                    result = Result()
                    result.evaluate(outputs.data, targets.data)

                    # Save some images for Comet
                    if i in img_idxs:
                        utils.log_image_to_comet(inputs[0], targets[0], outputs[0], epoch, i, experiment, "val")

                    experiment.log_metric("val_loss", val_loss.item(), step=val_step)
                    experiment.log_metric("rmse", result.rmse, step=train_step)
                    experiment.log_metric("mae", result.mae, step=train_step)
                    experiment.log_metric("delta1", result.delta1, step=train_step)

                    val_step += 1

                # Mean validation loss
                mean_val_loss = running_val_loss / len(val_loader)
                print("Validation Loss [%d]: %.3f" % (current_epoch, mean_val_loss))

            # Save periodically
            if (epoch + 1) % save_frequency == 0:
                save_path = utils.get_save_path(current_epoch, experiment_dir)
                utils.save_model(model, optimizer, save_path, current_epoch, best_loss, max_checkpoints)
                print("Saving new checkpoint")

    print("Finished training")

    # Save the final model
    save_path = utils.get_save_path(num_epochs, experiment_dir)
    utils.save_model(model, optimizer, save_path, current_epoch, mean_val_loss, max_checkpoints)
    print("Model saved to ", os.path.abspath(save_path))

    with experiment.test():
        model.eval()
        running_loss = 0.0
        average_meter = AverageMeter()
        img_idxs = np.random.randint(0, len(test_loader), size=min(len(test_loader), 50))
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Predict
            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            result = Result()
            result.evaluate(outputs.data, targets.data)
            average_meter.update(result, 0, 0, inputs.size(0))

            # Log images to comet
            if i in img_idxs:
                utils.log_image_to_comet(inputs[0], targets[0], outputs[0], 0, i, experiment, "test")

        # Mean validation loss
        mean_loss = running_loss / len(test_loader)
        print("Average Test Loss: %.3f" % (mean_loss))

        average = average_meter.average()
        experiment.log_metric("loss", mean_loss)
        experiment.log_metric("average_rmse", average.rmse)
        experiment.log_metric("average_mae", average.mae)
        experiment.log_metric("average_delta1", average.delta1)

except KeyboardInterrupt:
    print("Saving model and quitting...")

    save_path = utils.get_save_path(current_epoch, experiment_dir)
    utils.save_model(model, optimizer, save_path, current_epoch, mean_val_loss, max_checkpoints)
    print("Model saved to ", os.path.abspath(save_path))
