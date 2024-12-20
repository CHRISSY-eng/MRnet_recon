# dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F

class MRIDataset(Dataset):
    def __init__(self, root_dir, labels_files, phase, views= ('coronal', 'axial', 'sagittal'), transform=None, target_size=(32, 128, 128)):
        """
        Args:
            root_dir (str): Directory with all the MRI images.
            labels_file (str): Path to the CSV file with labels.
            phase (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Desired output size (C, D, H, W).
        """
        self.root_dir = os.path.join(root_dir, phase)
        self.phase = phase
        self.views = views
        self.transform = transform
        self.target_size = target_size
        
        self.index_offset = 1130 if phase == 'valid' else 0
        self.view_prefix = {
            'coronal': 'CORO',
            'axial': 'AXIA',
            'sagittal': 'SAGI'
        }

        if phase == 'train':
            self.labels_abnormal = pd.read_csv(labels_files['abnormal'], header=0)
        else:
            self.labels_abnormal = pd.read_csv(labels_files['abnormal'], header=None, names=['case', 'Label'])

        self.labels_acl = pd.read_csv(labels_files['acl'], header=None, names=['case', 'Label'])
        self.labels_meniscus = pd.read_csv(labels_files['meniscus'], header=None, names=['case', 'Label'])

        self.labels_abnormal.sort_values('case', inplace=True)
        self.labels_acl.sort_values('case', inplace=True)
        self.labels_meniscus.sort_values('case', inplace=True)
        self.labels_abnormal.reset_index(drop=True, inplace=True)
        self.labels_acl.reset_index(drop=True, inplace=True)
        self.labels_meniscus.reset_index(drop=True, inplace=True)

        #self.max_samples = 10

    def __len__(self):
        return len(self.labels_abnormal)

    def __getitem__(self, idx):
        #view_idx = idx % len(self.views)
        actual_idx = idx
        #view = self.views[view_idx]
        view = self.views[0]
        image_idx = actual_idx + self.index_offset

        view_dir = os.path.join(self.root_dir, view)
        prefix = self.view_prefix[view]
        image_name = f"{prefix}_000_{image_idx:04d}.nii.gz"
        image_path = os.path.join(view_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found!")
            return torch.zeros(1, *self.target_size), torch.zeros(3)
        # Normalize intensity values
        try:
            
            image = nib.load(image_path).get_fdata()
            image = torch.tensor(image).unsqueeze(0).float()
            image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(1, 1, 1), torch.zeros(3) 


        label_abnormal = float(self.labels_abnormal.iloc[actual_idx]['Label'])
        label_acl = float(self.labels_acl.iloc[actual_idx]['Label'])
        label_meniscus = float(self.labels_meniscus.iloc[actual_idx]['Label'])

        labels = torch.tensor([label_abnormal, label_acl, label_meniscus], dtype=torch.float32)

        return image, labels
