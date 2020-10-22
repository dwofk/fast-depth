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

    # Data loading code
    print("=> creating data loaders...")
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
    print("=> data loaders created.")

    print("=> loading model '{}'".format(args.model))
    model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)

    # Set GPU
    device = torch.device("cuda:{}".format(args.gpu)
                          if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print("Using device", device)

    # Load the checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if type(checkpoint) is dict:
        state_dict = checkpoint['model_state_dict']
        args.start_epoch = checkpoint['epoch']
        print("=> loaded epoch {}".format(checkpoint['epoch']))
    else:
        state_dict = checkpoint
        # model = checkpoint
        args.start_epoch = 0

    # create new OrderedDict that does not contain `module.`
    state_dict = utils.convert_state_dict_from_gpu(state_dict)

    # load params
    model.load_state_dict(state_dict)

    output_directory = os.path.join(os.path.dirname(args.model), "images")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Saving results to " + output_directory)

    validate(val_loader, model, args.start_epoch, device,
             output_directory, args.num_photos_saved, args.write)


def validate(loader, model, epoch, device, output_directory, num_photos, write_to_file):
    average_meter = AverageMeter()

    # Randomly choose some results to save
    idx_to_save = np.random.randint(low=0, high=len(loader), size=num_photos)

    model.eval()
    model.to(device)

    end = time.time()
    for i, (input, target) in enumerate(loader):
        input, target = input.to(device), target.to(device)
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save images for visualization
        if i in idx_to_save:
            img_merge = utils.merge_into_row(input, target, pred)
            utils.write_results(img_merge, result)
            filename = os.path.join(output_directory, \
                "comparison_epoch_{}_{}.png".format(str(epoch), np.where(idx_to_save == i)[0][0]))
            utils.save_image(img_merge, filename)

        # if j < num_photos:
        #     if (idx_start + 8 * skip) <= len(loader):
        #         if i == idx_start:
        #             img_merge = utils.merge_into_row(input, target, pred)
        #         elif (i < 8*skip + idx_start) and (i % skip == 0):
        #             row = utils.merge_into_row(input, target, pred)
        #             img_merge = utils.add_row(img_merge, row)
        #         elif i == 8*skip + idx_start:
        #             filename = output_directory + '/comparison_epoch_' + \
        #                 str(epoch) + '_' + str(j) + '.png'
        #             utils.save_image(img_merge, filename)
        #             idx_start += 9*skip
        #             j += 1

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                      i+1, len(loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(
              average=avg, time=avg.gpu_time))

    if write_to_file:
        filename = os.path.join(output_directory, "results.csv")
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                             'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                             'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge


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
