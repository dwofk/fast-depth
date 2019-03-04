import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size

class NYUDataset(MyDataloader):
    def __init__(self, root, split, modality='rgb'):
        self.split = split
        super(NYUDataset, self).__init__(root, split, modality)
        self.output_size = (224, 224)

    def is_image_file(self, filename):
        # IMG_EXTENSIONS = ['.h5']
        if self.split == 'train':
            return (filename.endswith('.h5') and \
                '00001.h5' not in filename and '00201.h5' not in filename)
        elif self.split == 'holdout':
            return ('00001.h5' in filename or '00201.h5' in filename)
        elif self.split == 'val':
            return (filename.endswith('.h5'))
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                "Supported dataset splits are: train, val"))

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((228, 304)),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np
