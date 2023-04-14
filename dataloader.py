import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import torch.distributed as dist
import re
import random

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

class NTIREDataset(torch.utils.data.Dataset):
    def __init__(self, dir="", mode="test"):
        super().__init__()
        dir = os.path.join(dir, mode)

        self.dir = dir
        self.gt_available = False

        input_names = []
        gt_names = []

        self.input_names = input_names
        self.gt_names = gt_names
        
        for root, dirs, files in os.walk(os.path.join(self.dir, 'hazy'), topdown=False):
            for name in files:
                input_names.append(os.path.join(root, name))

        if mode == "train" or mode== "val":
            self.gt_available = True
            for root, dirs, files in os.walk(os.path.join(self.dir, 'GT'), topdown=False):
                for name in files:
                    gt_names.append(os.path.join(root, name))

    def get_images(self, index):
        input_name = self.input_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        # read image, and scale [0, 1] to [-1, 1]
        if self.gt_available :
            gt_name = self.gt_names[index]
            try:
                gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
            except:
                gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                    PIL.Image.open(gt_name).convert('RGB')
            
            if self.gt_available:
                return transforms(input_img)* 2 - 1, transforms(gt_img)* 2 - 1, img_id
            else:
                return transforms(input_img)* 2 - 1, img_id
        else:
            return transforms(input_img), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
