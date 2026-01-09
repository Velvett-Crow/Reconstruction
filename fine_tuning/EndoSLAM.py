"""
EndoSLAM dataset loader formatted for Depth Anything V2 fine-tuning.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet


class EndoSLAM(Dataset):

    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size

        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def __getitem__(self, item):
        img_path, depth_path = self.filelist[item].split(' ')

        # Load RGB
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # [0,1]

        # Load Depth PNG
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # Extract R channel
        if len(depth_png.shape) == 3:
            depth_raw = depth_png[:, :, 2]
        else:
            depth_raw = depth_png  # single-channel fallback

        depth = depth_raw.astype('float32')

        # Convert depth to meters (8 bit)
        depth = (depth / 255) * 0.001

        # Apply project transforms
        sample = self.transform({'image': image, 'depth': depth})

        # Convert to torch tensors
        sample['image'] = torch.from_numpy(sample['image'])        # [3,H,W]
        sample['depth'] = torch.from_numpy(sample['depth'])        # [H,W]

        # Valid mask: depth > 0
        sample['valid_mask'] = sample['depth'] > 0
        sample['image_path'] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)
