import numpy as np
import matplotlib as mp

mp.use("pdf")
import matplotlib.pyplot as plt

cmap = plt.cm.viridis

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # HWC

def save_rgb_image(rgb_npy_fp, filename):
    rgb_np = np.load(rgb_npy_fp)
    rgb_scaled = 255 * np.transpose(np.squeeze(rgb_np), (0,1,2)) # HWC
    mp.image.imsave('rgb.png', rgb_scaled.astype('uint8'))

def save_depth_image(depth_npy_fp, filename):
    depth_np = np.load(depth_npy_fp)
    depth_np_color = colored_depthmap(depth_np)
    mp.image.imsave('depth.png', depth_np_color.astype('uint8'))

def save_pred_image(pred_npy_fp, filename):
    pred_np = np.load(pred_npy_fp)
    pred_np_2d = pred_np[0,0,:,:] # HW
    pred_np_color = colored_depthmap(pred_np_2d)
    mp.image.imsave('pred.png', pred_np_color.astype('uint8'))

save_rgb_image('rgb.npy', 'rgb.png')
save_depth_image('depth.npy', 'depth.png')
save_pred_image('pred.npy', 'pred.png')
