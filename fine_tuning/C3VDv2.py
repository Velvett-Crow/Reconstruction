"""
C3VDv2 dataset loader for Depth Anything V2 (validation).
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet


class C3VDv2(Dataset):

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
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            PrepareForNet(),
        ])

    def __getitem__(self, item):
        img_path, depth_path = self.filelist[item].split(' ')

        # Load RGB
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Load Depth (16-bit TIFF)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')

        # Convert TIFF to meters: (0–65535) -> (0–0.1 m)
        depth = (depth_raw / 65535.0) * 0.1

        valid_mask = (depth > 0.0)

        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])      # [3,H,W]
        sample['depth'] = torch.from_numpy(sample['depth'])      # [H,W]
        sample['valid_mask'] = torch.from_numpy(valid_mask).float()  # [H,W]
        sample['image_path'] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)
