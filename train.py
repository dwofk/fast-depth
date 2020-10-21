import os
import sys
import json
import argparse
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


def get_params(file):
    params = utils.load_training_parameters(params_file)

    # Convert from JSON format to DataLoader format
    params["training_dataset_paths"] = utils.format_dataset_path(
        params["training_dataset_paths"])
    params["test_dataset_paths"] = utils.format_dataset_path(
        params["test_dataset_paths"])
    return params


def set_up_experiment(params, experiment, resume=None):

    # Log hyper params to Comet
    hyper_params = {
        "learning_rate": params["optimizer"]["lr"],
        "momentum": params["optimizer"]["momentum"],
        "weight_decay": params["optimizer"]["weight_decay"],
        "optimizer": params["optimizer"]["type"],
        "loss": params["loss"],
        "num_epochs": params["num_epochs"],
        "batch_size": params["batch_size"],
        "train_val_split": params["train_val_split"][0],
        "depth_max": params["depth_max"]
    }
    experiment.log_parameters(hyper_params)
    experiment.add_tag(params["loss"])

    # Log dataset info to Comet
    training_folders = ", ".join(params["training_dataset_paths"])
    test_folders = ", ".join(params["test_dataset_paths"])
    experiment.log_dataset_info(path=training_folders)
    experiment.log_other("test_dataset_info", test_folders)

    train_loader, val_loader, test_loader = load_dataset(params)

    # Configure GPU
    params["device"] = torch.device("cuda:{}".format(params["device"]) if type(
        params["device"]) is int and torch.cuda.is_available() else "cpu")

    model, optimizer_state_dict = load_model(params, resume)

    # Create experiment directory
    if resume:
        experiment_dir = os.path.split(resume)[0]  # Use existing folder
    else:
        experiment_dir = utils.make_dir_with_date(
            params["save_dir"], "fastdepth")  # New folder
    print("Saving results to ", experiment_dir)
    params["experiment_dir"] = experiment_dir

    # Use parallel GPUs if available
    # Specify which GPUs to use on DGX
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if os.environ["USE_MULTIPLE_GPUS"] == "TRUE" and torch.cuda.device_count() > 1:
            print("Let's use", num_gpus, "GPUs!")
            model = nn.DataParallel(model)
    except KeyError:
        pass

    # Send model to GPU(s)
    # This must be done before optimizer is created if a model state_dict is being loaded
    model.to(params["device"])

    # Loss & Optimizer
    if params["loss"] == "L2":
        criterion = torch.nn.MSELoss()
        print("Using L2 Loss")
    else:
        criterion = torch.nn.L1Loss()
        print("Using L1 Loss")

    if params["optimizer"]["type"] == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=params["optimizer"]["lr"],
                              momentum=params["optimizer"]["momentum"],
                              weight_decay=params["optimizer"]["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=params["optimizer"]["lr"])

    experiment.add_tag(params["optimizer"]["type"])
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    # Load optimizer tensors onto GPU if necessary
    utils.optimizer_to_gpu(optimizer)

    return params, train_loader, val_loader, test_loader, model, criterion, optimizer


def load_dataset(params):
    # Create dataset
    print("Loading the dataset...")
    dataset = Datasets.FastDepthDataset(params["training_dataset_paths"],
                                        split='train',
                                        depthMin=params["depth_min"],
                                        depthMax=params["depth_max"],
                                        input_shape_model=(224, 224))

    test_dataset = Datasets.FastDepthDataset(params["test_dataset_paths"],
                                             split='val',
                                             depthMin=params["depth_min"],
                                             depthMax=params["depth_max"],
                                             input_shape_model=(224, 224))

    # Make training/validation split
    train_val_split_lengths = utils.get_train_val_split_lengths(
        params["train_val_split"], len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, train_val_split_lengths)
    print("Train/val split: ", train_val_split_lengths)
    params["num_training_examples"] = len(train_dataset)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params["batch_size"],
                                               shuffle=True,
                                               num_workers=params["num_workers"],
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=params["batch_size"],
                                             shuffle=True,
                                             num_workers=params["num_workers"],
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=params["batch_size"],
                                              shuffle=False,
                                              num_workers=params["num_workers"],
                                              pin_memory=True)

    return train_loader, val_loader, test_loader


def load_model(params, resume=None):
    # Load model checkpoint if specified
    model_state_dict,\
        optimizer_state_dict,\
        params["start_epoch"], _ = utils.load_checkpoint(resume)
    model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)

    # Load the model
    model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    return model, optimizer_state_dict
    

def train(params, train_loader, val_loader, model, criterion, optimizer, experiment):
    mean_val_loss = -1
    try:
        train_step = 0
        val_step = 0
        for epoch in range(params["num_epochs"] - params["start_epoch"]):
            current_epoch = params["start_epoch"] + epoch + 1

            epoch_loss = 0.0
            running_loss = 0.0
            average = AverageMeter()
            img_idxs = np.random.randint(0, len(train_loader), size=5)

            model.train()
            with experiment.train():
                for i, (inputs, targets) in enumerate(train_loader):
                    # Send data to GPU
                    inputs, targets = inputs.to(
                        params["device"]), targets.to(params["device"])

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Predict
                    outputs = model(inputs)

                    # Loss and backprop
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # Calculate metrics
                    result = Result()
                    result.evaluate(outputs.data, targets.data)
                    average.update(result, 0, 0, inputs.size(0))
                    epoch_loss += loss.item()

                    # Log to Comet
                    utils.log_comet_metrics(
                        experiment, result, loss.item(), step=train_step, epoch=current_epoch)
                    train_step += 1

                    # Log images to Comet
                    if i in img_idxs:
                        utils.log_image_to_comet(
                            inputs[0], targets[0], outputs[0], epoch, i, experiment, "train")

                    # Print statistics
                    running_loss += loss.item()
                    if (i + 1) % params["stats_frequency"] == 0 and i != 0:
                        print('[%d, %5d] loss: %.3f' %
                              (current_epoch, i + 1, running_loss / params["stats_frequency"]))
                        running_loss = 0.0

                # Log epoch metrics to Comet
                mean_train_loss = epoch_loss/len(train_loader)
                utils.log_comet_metrics(experiment, average.average(), mean_train_loss,
                                        prefix="epoch", step=train_step, epoch=current_epoch)

            # Validation each epoch
            epoch_loss = 0.0
            average = AverageMeter()
            with experiment.validate():
                with torch.no_grad():
                    img_idxs = np.random.randint(0, len(val_loader), size=5)
                    model.eval()
                    for i, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(
                            params["device"]), targets.to(params["device"])

                        # Predict
                        outputs = model(inputs)

                        # Loss
                        loss = criterion(outputs, targets)

                        # Calculate metrics
                        result = Result()
                        result.evaluate(outputs.data, targets.data)
                        average.update(result, 0, 0, inputs.size(0))
                        epoch_loss += loss.item()

                        # Log to Comet
                        utils.log_comet_metrics(
                            experiment, result, loss.item(), step=val_step, epoch=current_epoch)
                        val_step += 1

                        # Log images to Comet
                        if i in img_idxs:
                            utils.log_image_to_comet(
                                inputs[0], targets[0], outputs[0], epoch, i, experiment, "val")

                    # Log epoch metrics to Comet
                    mean_val_loss = epoch_loss / len(val_loader)
                    utils.log_comet_metrics(experiment, average.average(), mean_val_loss,
                                            prefix="epoch", step=val_step, epoch=current_epoch)
                    print("Validation Loss [%d]: %.3f" %
                          (current_epoch, mean_val_loss))

            # Save periodically
            if (epoch + 1) % params["save_frequency"] == 0:
                save_path = utils.get_save_path(
                    current_epoch, params["experiment_dir"])
                utils.save_model(model, optimizer, save_path, current_epoch,
                                 mean_val_loss, params["max_checkpoints"])
                experiment.log_model(save_path.split("/")[-1], save_path)
                print("Saving new checkpoint")

            experiment.log_epoch_end(current_epoch)

        print("Finished training")

        # Save the final model
        save_path = utils.get_save_path(
            params["num_epochs"], params["experiment_dir"])
        utils.save_model(model, optimizer, save_path, current_epoch,
                         mean_val_loss, params["max_checkpoints"])
        experiment.log_model(save_path.split("/")[-1], save_path)
        print("Model saved to ", os.path.abspath(save_path))

    except KeyboardInterrupt:
        print("Saving model and quitting...")
        save_path = utils.get_save_path(
            current_epoch, params["experiment_dir"])
        utils.save_model(model, optimizer, save_path, current_epoch,
                         mean_val_loss, params["max_checkpoints"])
        experiment.log_model(save_path.split("/")[-1], save_path)
        print("Model saved to ", os.path.abspath(save_path))


def evaluate(params, loader, model, criterion, experiment):
    print("Testing...")
    with experiment.test():
        running_loss = 0.0
        total_loss = 0.0
        average = AverageMeter()
        img_idxs = np.random.randint(0, len(loader), size=min(len(loader), 50))
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(
                params["device"]), targets.to(params["device"])

            # Predict
            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            result = Result()
            result.evaluate(outputs.data, targets.data)
            average.update(result, 0, 0, inputs.size(0))

            # Log images to comet
            if i in img_idxs:
                utils.log_image_to_comet(
                    inputs[0], targets[0], outputs[0], 0, i, experiment, "test")

            running_loss += loss.item()
            if (i + 1) % params["stats_frequency"] == 0 and i != 0:
                print('[{}/{}] loss: {:.3f}'.format(i + 1, len(loader),
                                                    running_loss / params["stats_frequency"]))
                running_loss = 0.0

        # Mean validation loss
        mean_loss = total_loss / len(loader)
        utils.log_comet_metrics(experiment, average.average(), mean_loss)
        print("Average Test Loss: %.3f" % (mean_loss))


def main(args):
    os.environ["USE_MULTIPLE_GPUS"] = "TRUE"

    # Create Comet experiment
    experiment = Experiment(
        api_key="Bq3mQixNCv2jVzq2YBhLdxq9A", project_name="fastdepth")

    if (args.tag):
        experiment.add_tag(args.tag)
    
    params = get_params(params_file)

    params, train_loader, val_loader, test_loader, \
        model, criterion, optimizer = set_up_experiment(
            params, experiment, args.resume)

    train(params, train_loader, val_loader,
          model, criterion, optimizer, experiment)

    evaluate(params, test_loader, model, criterion, experiment)


if __name__ == "__main__":
    # Parse command args
    parser = argparse.ArgumentParser(description='FastDepth Training')
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to model checkpoint to resume training.")
    parser.add_argument('-t', '--tag', type=str, default=None,
                        help='Extra tag to add to Comet experiment')
    args = parser.parse_args()
    main(args)
