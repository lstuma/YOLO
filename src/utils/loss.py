import torch
import numpy as np
from math import sqrt

class YOLOLoss:

    def __init__(self, n_classes, n_grid_cells, w_coord=1., w_conf=1., w_class=1., w_noobj=.5, noobj_threshold=.5):
        """
        n_classes: int
        n_grid_cells: int
        w_coord: float
        w_conf: float
        w_class: float
        w_noobj: float
        noobj_threshold: float
        """
        self.C = n_classes
        self.S = n_grid_cells
        self.w_coord = w_coord
        self.w_conf = w_conf
        self.w_class = w_class
        self.w_noobj = w_noobj
        self.noobj_threshold = noobj_threshold

    def __call__(self, x, y):
        """
        x: torch.Tensor = [[pc, bx, by, bh, bw, c1, c2,..., cn],]
        y: torch.Tensor = [[pc, bx, by, bh, bw, c1, c2,..., cn],]
        """

        return    self.w_coord * self._localization_loss(x, y)  \
               +  self.w_conf  * self._confidence_loss(x, y)    \
               +  self.w_class * self._class_loss(x, y)


    def _localization_loss(self, x, y):
        return sum(sum((
                            (x[i][1] - y[i][1])**2,
                            (x[i][2] - y[i][2])**2,
                            (sqrt(x[i][3]) - sqrt(y[i][3]))**2,
                            (sqrt(x[i][4]) - sqrt(y[i][4]))**2,
                        )) for i in range((self.S**2)+1) if y[i][0] > self.noobj_threshold)

    def _confidence_loss(self, x, y):
        obj = sum((x[i][0] - x[i][0])**2 for i in range((self.S**2)+1) if y[i][0] > self.noobj_threshold)

        noobj = sum((x[i][0] - x[i][0])**2 for i in range((self.S**2)+1) if y[i][0] <= self.noobj_threshold)

        return obj + self.w_noobj * noobj

    def _class_loss(self, x, y):
        return sum(sum((x[4+c]-y[4+c])**2 for c in range(1, self.C+1)) for i in range((self.S**2)+1) if y[i][0] > self.noobj_threshold)

