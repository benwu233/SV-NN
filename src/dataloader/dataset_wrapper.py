
import torch
from torch.utils.data import Dataset

class mydata(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, i):
        x = self.x_list[i].reshape(-1)
        w = torch.tensor([1.])
        y = torch.tensor([self.y_list[i]])
        return (x, w), y
