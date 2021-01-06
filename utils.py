import os
import cv2
import torch
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import datetime
from collections import OrderedDict
import models

cmap = plt.cm.viridis

# Here for legacy code


def parse_command():
    data_names = ['nyudepthv2',
                  'unreal']

    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--gpu', default='0', type=str,
                        metavar='N', help="gpu id")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to model checkpoint to resume training.")
    parser.add_argument('-n', '--num_photos_saved', type=int, default=1,
                        help="Number of comparison photos to save during evaluation.")
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def visualize_depth(depth, _min=None, _max=None):
    # so the image isn't all white, convert it to range [0, 1.0]
    _mean, _std = (np.mean(depth), np.std(depth))

    if _min is None:
        _min = np.min(depth)
    if _max is None:
        _max = np.max(depth)

    newMax = _mean + 2 * _std
    newMin = _mean - 2 * _std
    if newMax < _max:
        _max = newMax
    if newMin > _min:
        _min = newMin
    _range = _max-_min
    if _range:
        depth -= _min
        depth /= _range

    # Convert to bgr
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    # Color mapping for better visibility / contrast
    depth = np.array(depth * 255, dtype=np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    return depth


def visualize_depth_compare(depth, target):
    _mean_target, _std_target = (np.mean(target), np.std(target))
    _min = min(np.min(target), np.min(depth))
    _max = max(np.max(target), np.max(depth))

    _range = _max-_min
    if _range:
        depth -= _min
        depth /= _range
        target -= _min
        target /= _range

    # Convert to bgr
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    # Color mapping for better visibility / contrast
    depth = np.array(depth * 255, dtype=np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    target = np.array(target * 255, dtype=np.uint8)
    target = cv2.applyColorMap(target, cv2.COLORMAP_TURBO)
    return depth, target


def merge_into_row(input, depth_target, depth_pred):
    # H, W, C
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    depth_pred_col, depth_target_col = visualize_depth_compare(
        depth_pred_cpu, depth_target_cpu)

    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    # H, W, C
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(
        depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(
        depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack(
        [rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def write_results(img, results):
    font = cv2.FONT_HERSHEY_SIMPLEX

    blank = np.zeros(shape=(224, 224, 3), dtype=np.float32)
    out = cv2.hconcat([img, blank])

    rmse = "RMSE: {:.2f}m".format(results.rmse)
    mae = "MAE: {:.2f}m".format(results.mae)
    delta1 = "Delta1: {:.2f}m".format(results.delta1)
    cv2.putText(out, rmse, (out.shape[1] - blank.shape[1], 30),
                font, 1, (255, 255, 255), 1)
    cv2.putText(out, mae, (out.shape[1] - blank.shape[1], 60),
                font, 1, (255, 255, 255), 1)
    cv2.putText(out, delta1, (out.shape[1] - blank.shape[1], 90),
                font, 1, (255, 255, 255), 1)

    return out


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def load_config_file(file):
    if not os.path.isfile(file):
        raise ValueError("Parameters file does not exist")

    return json.load(open(file))


def format_dataset_path(dataset_paths):
    if isinstance(dataset_paths, str):
        data_paths = {
            dataset_paths
        }
    elif isinstance(dataset_paths, list):
        data_paths = set()
        for path in dataset_paths:
            data_paths.add(path)

    return data_paths


def make_dir_with_date(root_dir, prefix):
    time = datetime.datetime.now()

    pid = None
    try:
        pid = os.environ["COMET_OPTIMIZER_PROCESS_ID"]
    except KeyError:
        pass

    date_dir = os.path.join(root_dir, prefix + "_" +
                            time.strftime("%m_%d_%H_%M"))
    if pid:
        date_dir += "_opt_{}".format(pid)

    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    return date_dir


def get_train_val_split_lengths(train_val_split, dataset_length):
    return [int(np.around(train_val_split[0] * 0.01 * dataset_length)),
            int(np.around(train_val_split[1] * 0.01 * dataset_length))]


def load_model(params, resume=None):
    # Load model checkpoint if specified
    model_state_dict,\
        optimizer_state_dict,\
        params["start_epoch"], _ = load_checkpoint(resume)
    model_state_dict = convert_state_dict_from_gpu(model_state_dict)

    # Load the model
    model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    return model, optimizer_state_dict


def load_checkpoint(model_path):
    if model_path and os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))

        checkpoint = torch.load(model_path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_result']

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))

    else:
        model_state_dict = None
        optimizer_state_dict = None
        start_epoch = 0
        best_loss = 100000  # Very high number

        if model_path:
            print("=> no checkpoint found at '{}'".format(model_path))

    return model_state_dict,\
        optimizer_state_dict,\
        start_epoch,\
        best_loss,\


def get_save_path(epoch, save_dir="./results"):
    save_path = os.path.join(
        save_dir, "model_{}.pth".format(str(epoch).zfill(4)))
    return save_path


def save_model(model, optimizer, save_path, epoch, loss, max_checkpoints=None):
    if max_checkpoints:
        checkpoint_dir = os.path.split(save_path)[0]
        checkpoints = sorted(os.listdir(checkpoint_dir))
        while len(checkpoints) >= max_checkpoints:
            # Remove oldest checkpoints
            os.remove(os.path.join(checkpoint_dir, checkpoints[0]))
            checkpoints.pop(0)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_result": loss
    }, save_path)


def optimizer_to_gpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.device == "cpu":
                state[k] = v.cuda()


def convert_state_dict_from_gpu(state_dict):
    if state_dict:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if ("module" in k):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        return new_state_dict
    else:
        return state_dict


def save_losses_plot(path, num_epochs, losses, title):
    x = np.arange(1, num_epochs + 1, 1)
    plt.plot(x, losses)
    plt.xticks(x, x)
    plt.xlabel("Epochs")
    plt.ylabel("{} Loss (m)".format(title))
    plt.title("{} Loss".format(title))
    plt.savefig(path, bbox_inches="tight")


def log_comet_metrics(experiment, result, loss, prefix=None, step=None, epoch=None):
    metrics = {
        "loss": loss,
        "rmse": result.rmse,
        "mae": result.mae,
        "delta1": result.delta1
    }
    experiment.log_metrics(metrics, prefix=prefix, step=step, epoch=epoch)


def log_image_to_comet(input, target, output, epoch, id, experiment, result, prefix, step=None):
    img_merge = merge_into_row(input, target, output)
    img_merge = write_results(img_merge, result)
    log_merged_image_to_comet(img_merge, epoch, id, experiment, prefix, step)


def stack_images(input, target, output):
    b, g, r = cv2.split(input)
    return cv2.merge((b, g, r, output, target))


def create_raw_image(image):
    return image.tobytes()    


def log_raw_image_to_comet(input, target, output, epoch, id, experiment, prefix, step=None):
    # C, H, W -> H, W, C
    input = np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    target = np.squeeze(target.cpu().numpy())
    output = np.squeeze(output.data.cpu().numpy())

    stacked_images = stack_images(input, target, output)
    raw_image = create_raw_image(stacked_images)
    log_merged_raw_image_to_comet(raw_image, epoch, id, experiment, prefix, step)


def log_merged_raw_image_to_comet(raw_image, epoch, id, experiment, prefix, step=None):
    file_name = "{}_epoch_{}_id_{}.raw".format(prefix, epoch, id)
    if step:
        step = int(step)

    experiment.log_asset_data(raw_image, name=file_name, step=step)


def log_merged_image_to_comet(img_merge, epoch, id, experiment, prefix, step=None):
    img_name = "{}_epoch_{}_id_{}".format(prefix, epoch, id)
    if step:
        step = int(step)
    experiment.log_image(img_merge, name=img_name, step=step)


def flip_depth(outputs, targets, clip=None):
    targets = (1 / targets)
    if clip:
        outputs[outputs < clip] = clip
    outputs = (1 / outputs)
    return outputs, targets


def process_for_loss(outputs, targets, predict_disparity, loss_disparity, disparity_constant, clip=0.1):
    c = disparity_constant if loss_disparity else 1
    if predict_disparity != loss_disparity:
        outputs, targets = flip_depth(outputs, targets, clip)

    return outputs, targets, c


def convert_to_depth(outputs, targets, not_clipped_yet, is_disparity, clip=None):
    clip = clip if not_clipped_yet else None
    if is_disparity:
        outputs, targets = flip_depth(outputs, targets, clip)

    return outputs, targets
