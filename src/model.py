import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.read_struct import read_struct


class YOLOModel(nn.Module):
    def __init__(self, path):
        super().__init__()

        with open(path, "r") as f:
            self.network = read_struct(f)


    def forward(self, x):
        x = self.network(x)
        return x



if __name__ == "__main__":
    y = YOLOModel("src/struct.nn")
    print(y)
    for i in y.network:
        if type(i) in (nn.Conv2d, nn.Linear):
            print(i.weight)
