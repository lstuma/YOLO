import torch
import torch.nn as nn
import torch.nn.functional as F


def read_struct(file):

    def _get_value(s):
        s = s.split("=")[1]
        return int(s)

    struct = []

    lst = [i.replace("\n", "").replace(",", "").split(" ") for i in file.readlines() if i != ""]

    for i in lst:
        match i[0]:
            case "CONV":
                struct.append(nn.Conv2d(_get_value(i[1]), _get_value(i[2]), _get_value(i[3]), stride=_get_value(i[4])))
            case "MAXP":
                struct.append(nn.MaxPool2d(_get_value(i[1]), stride=_get_value(i[2]), padding=_get_value(i[3])))
            case "LINL":
                struct.append(nn.Linear(_get_value(i[1]), _get_value(i[2])))
            case other:
                continue
        struct.append(nn.ReLU())

    return nn.Sequential(*struct)
