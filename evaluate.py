from comet_ml import Experiment, ExistingExperiment
import utils
from metrics import AverageMeter, Result
import models
import os
import sys
import time
import csv
import numpy as np
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

#dataset_path = "/workspace/mnt/repositories/bayesian-visual-odometry/scripts"
dataset_path = "/workspace/data/alex/bayesian-visual-odometry/scripts"
sys.path.append(dataset_path)
import Datasets

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
              'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']

def main(args):

    if args.experiment_key:
        experiment = ExistingExperiment(api_key="jBFVYFo9VUsy0kb0lioKXfTmM", previous_experiment=args.experiment_key)
    else:
        experiment = Experiment(api_key="jBFVYFo9VUsy0kb0lioKXfTmM", project_name="fastdepth")

    params = {
        "experiment_key" : args.experiment_key,
        "save_image" : args.save_images,
        "stats_frequency" : args.print_freq,
        "save_metrics" : args.save_metrics,
    }

    # Data loading code
    print("Creating data loaders...")
    if args.data == 'nyudepthv2':
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(args.directory, split='val')
    elif args.data == "unreal":
        val_dataset = Datasets.FastDepthDataset(
            { args.directory }, split='val', input_shape_model=(224, 224), depthMax = args.max_depth)

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True)

    print("Loading model '{}'".format(args.model))
    if args.data == "unreal":
        model, _ = utils.load_model(params, args.model)
    else:
        # Maintain compatibility for fastdepth NYU model format
        state_dict = torch.load(args.model, map_location=params["device"])
        model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
        model.load_state_dict(state_dict)
        params["start_epoch"] = 0

    # Set GPU
    params["device"] = torch.device("cuda:{}".format(args.gpu)
                          if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print("Using device", params["device"])


    # Create output directory
    output_directory = os.path.join(os.path.dirname(args.model), "images")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    params["experiment_dir"] = output_directory
    print("Saving results to " + output_directory)

    evaluate(params, val_loader, model, experiment)


def evaluate(params, loader, model, experiment):
    print("Testing...")
    with experiment.test():
        with torch.no_grad():
            average = AverageMeter()
            img_idxs = np.random.randint(0, len(loader), size=min(len(loader), 50))
            for i, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(
                    params["device"]), targets.to(params["device"])

                data_time = time.time() - end

                # Predict
                end = time.time()
                outputs = model(inputs)
                gpu_time = time.time() - end

                result = Result()
                result.evaluate(outputs.data, targets.data)
                average.update(result, gpu_time, data_time, inputs.size(0))
                end = time.time()

                # Log images to comet
                if i in img_idxs:
                    img_merge = utils.merge_into_row(inputs[0], targets[0], outputs[0])
                    if params["experiment_key"]:
                        utils.log_merged_image_to_comet(img_merge, 0, i, experiment, "test")
                    if params["save_image"]:                    
                        filename = os.path.join(params["experiment_dir"], \
                            "comparison_epoch_{}_{}.png".format(str(params["start_epoch"]), np.where(img_idxs == i)[0][0]))
                        utils.save_image(img_merge, filename)

                if (i + 1) % params["stats_frequency"] == 0 and i != 0:
                    print('Test: [{0}/{1}]\t'
                          't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                          'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                          'MAE={result.mae:.2f}({average.mae:.2f}) '
                          'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                          'REL={result.absrel:.3f}({average.absrel:.3f}) '
                          'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                              i+1, len(loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

            # Mean validation loss
            avg = average.average()
            utils.log_comet_metrics(experiment, avg, None)
            print('\n*\n'
                  'RMSE={average.rmse:.3f}\n'
                  'MAE={average.mae:.3f}\n'
                  'Delta1={average.delta1:.3f}\n'
                  'REL={average.absrel:.3f}\n'
                  'Lg10={average.lg10:.3f}\n'    
                  't_GPU={time:.3f}\n'.format(average=avg, time=avg.gpu_time))

            if params["save_metrics"]:
                filename = os.path.join(params["experiment_dir"], "results.csv")
                with open(filename, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                                    'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                                    'data_time': avg.data_time, 'gpu_time': avg.gpu_time})


if __name__ == '__main__':
    data_names = ['nyudepthv2',
                  'unreal']

    parser = argparse.ArgumentParser(description='FastDepth evaluation')
    parser.add_argument('-m', '--model', default=str, help="Path to model.")
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('-d', '--directory', type=str,
                        help="Directory of images to evaluate.")
    parser.add_argument('-e', '--experiment_key', type=str, default=None, help="Comet ML Experiment key.")
    parser.add_argument('--save_images', type=bool, default=False, help= "Save images to file.")
    parser.add_argument('--save_metrics', type=bool, default=False, help="Save metrics to file")
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--max-depth', type=int, default=25,
                        help="Maximum depth for ground truth.")
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--gpu', default=-1, type=int,
                        metavar='N', help="gpu id")
    parser.add_argument('-n', '--num_photos_saved', type=int, default=1,
                        help="Number of comparison photos to save during evaluation.")
    parser.add_argument('-w', '--write', default=False, help="Whether or not to write results to CSV.")

    args = parser.parse_args()

    assert os.path.isfile(args.model), \
        "=> no model found at '{}'".format(args.evaluate)

    main(args)
