from torch.utils.data import Dataset
import torch
import numpy as np

from typing import List


class Vin2ParamDataset(Dataset):
    def __init__(self, vins: List[np.ndarray], labels: np.ndarray):
        super(Vin2ParamDataset, self).__init__()
        self.vins = vins
        self.labels = labels

    def __getitem__(self, item):
        return torch.tensor(self.vins[item]), \
               torch.from_numpy(self.labels[item])

    def __len__(self):
        return len(self.vins)
