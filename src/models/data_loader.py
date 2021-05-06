# ./data_loader.py
""" Necessary classes to load the data into pytorch.

Available classes:
- KidneyStonesDataset
- ToTensor
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class KidneyStoneDataset(Dataset):
    """Kidney Stones dataset.
    First column is:  Size of kidney stone (if 1 L arge)
    Second column is: Treatment assigned (if 1 A)
    Third column is:  Recovery status (if 1 Recovered)
    The edges associated with the causal model are:
    L->T, L->R, T->R
    """

    def __init__(self, npy_file, bootstrap=None, transform=None, 
                        idx_mean=None, idx_sd=None, use_polynomials=False):
        """
        Args:
            npy_file (string): Path to the txt file with kidney stones.
            bootstrap (int): Random number generator to perform the bootstrap sample.
            transform: Transformation to be applied to the dataset.
            idx_norm: index of columns to be normalized, if any.
        """
        self.ks_dataset = np.load(npy_file)
        self.length_data = len(self.ks_dataset)
        self.transform = transform

        self.idx_mean = idx_mean
        self.idx_sd = idx_sd

        if use_polynomials:
            pre_polynomials = self.ks_dataset[:, :2]
            squares = pre_polynomials**2
            interactions = pre_polynomials[:,0]*pre_polynomials[:,1]
            self.ks_dataset = np.hstack((self.ks_dataset, squares, interactions.reshape(-1,1)))

        if bootstrap is not None:
            np.random.seed(bootstrap)
            self.ks_dataset = self.ks_dataset[np.random.choice(self.length_data, size=self.length_data),:]

        self.mean = np.asarray(np.mean(self.ks_dataset, axis=0))
        self.sd = np.asarray(np.std(self.ks_dataset, axis=0))

        # Substract the mean
        if idx_mean is not None:
            self.ks_dataset[:, self.idx_mean] -= self.mean[self.idx_mean]

        # Standarize the data
        if self.idx_sd is not None:
            self.ks_dataset[:, self.idx_sd] /= self.sd[self.idx_sd]

    def __len__(self):
        return self.length_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ks_dataset[idx, :]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()