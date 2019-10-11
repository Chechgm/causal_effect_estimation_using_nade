import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class KidneyStoneDataset(Dataset):
    """Kidney Stones dataset.
    First column is:  Size of kidney stone (if 1 Large)
    Second column is: Treatment assigned (if 1 A)
    Third column is:  Recovery status (if 1 recovered)
    The edges associated with the causal model are:
    L->T, L->R, T->R
    """

    def __init__(self, npy_file, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with kidney stones.
            root_dir (string): Directory with all the images.
        """
        self.ks_dataset = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return len(self.ks_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ks_dataset[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)
