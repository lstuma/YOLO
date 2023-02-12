import torch
import torch.nn as nn
import torch.nn.functional as F


def read_struct(file):

    def get_int(s):
        s = s.split("=")[1]
        return int(s)

    struct = []

    lst = [i.replace("\n", "").replace(",", "").split(" ") for i in file.readlines() if i != ""]

    for i in lst:
        match i[0]:
            case "CONV":
                struct.append(nn.Conv2d(get_int(i[1]), get_int(i[2]), get_int(i[3]), stride=get_int(i[4])))
            case "MAXP":
                struct.append(nn.MaxPool2d(get_int(i[1]), stride=get_int(i[2]), padding=get_int(i[3])))
            case "LINL":
                struct.append(nn.Linear(get_int(i[1]), get_int(i[2])))
            case other:
                continue
        struct.append(nn.ReLU())

    return nn.Sequential(*struct)
