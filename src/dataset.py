from torch.utils import data

class Dataset(data.Dataset):
    def __init__ (self, x, y, class_count=None):
        self.x = x
        self.y = y
        self.class_count = class_count

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
