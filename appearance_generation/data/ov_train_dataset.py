from torch.utils.data.dataset import Dataset

from data.image_folder import make_dataset

import os
from PIL import Image
from glob import glob as glob
import numpy as np
import random
import torch
from util import transforms


class RegularDataset(Dataset):
    def __init__(self, opt, augment):
        self.opt = opt
        self.root = opt.dataroot
        self.transforms = augment

        self.transform = transforms.Transforms([
            # transforms.SyncRandomHorizontalFlip(),
            # transforms.SyncRandomRotation((-5, 5)),
            # transforms.SyncRandomScaledCrop((1.0, 1.1))
        ])

        # input A (label maps)
        dir_A = '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        # input B (label images)
        dir_B = '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        img_A = Image.open(A_path)

        B_path = self.B_paths[index]
        img_B = Image.open(B_path)

        img_A, img_B = self.transform(img_A, img_B)

        # input A (label maps)
        A = self.parsing_embedding(img_A, 'seg')  # channel(20), H, W
        # A_tensor = self.transforms['1'](A)
        A_tensor = torch.from_numpy(A)

        # input B (images)
        B = np.array(img_B)
        B_tensor = self.transforms['1'](B)

        # original seg mask
        seg_mask = img_A
        seg_mask = np.array(seg_mask)
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)

        input_dict = {'seg_map': A_tensor, 'target': B_tensor, 'seg_map_path': A_path,
                      'target_path': B_path, 'seg_mask': seg_mask}

        return input_dict

    def parsing_embedding(self, parse, parse_type = "seg"):
        if parse_type == "seg":
            parse = np.array(parse)
            parse_channel = 20

        parse_emb = []
        for i in range(parse_channel):
            parse_emb.append((parse == i).astype(np.float32).tolist())
            
        parse = np.array(parse_emb).astype(np.float32)
        return parse  # (channel,H,W)

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'RegularDataset'
