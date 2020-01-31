import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class KidneyStoneDataset(Dataset):
    """Kidney Stones dataset.
    First column is:  Size of kidney stone (if 1 L arge)
    Second column is: Treatment assigned (if 1 A)
    Third column is:  Recovery status (if 1 Recovered)
    The edges associated with the causal model are:
    L->T, L->R, T->R
    """

    def __init__(self, npy_file, transform=None, idx_mean=None, idx_sd=None):
        """
        Args:
            npy_file (string): Path to the txt file with kidney stones.
            transform: Transformation to be applied to the dataset
            idx_norm: index of columns to be normalized, if any
        """
        self.ks_dataset = np.load(npy_file)
        self.transform = transform
        
        self.idx_mean = idx_mean
        self.idx_sd = idx_sd
        self.mean = torch.from_numpy(np.mean(self.ks_dataset, axis=0)).float()
        self.sd   = torch.from_numpy(np.std(self.ks_dataset, axis=0)).float()

    def __len__(self):
        return len(self.ks_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ks_dataset[idx, :]

        if self.transform:
            sample = self.transform(sample)
            
        # Substract the mean
        if self.idx_mean:
            sample[self.idx_mean] -= self.mean[self.idx_mean] 
            
        # Standarize the data
        if self.idx_sd:
            sample[self.idx_sd] /= self.sd[self.idx_sd]

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()
