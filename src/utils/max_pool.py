import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxPoolLayer(nn.Module):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def forward(self, X):
        n_c, n_x, n_y = tuple(X.size())
        X_new = np.zeros(n_c, n_x/self.n, n_y/self.n)
        for channel in X:
            for x in range(0, len(channel), self.n):
                for y in range(0, len(channel), self.n):
                    X_new[channel, x/n, y/n] = [
                        channel[x][y],
                        ch
                    ]





        return X
