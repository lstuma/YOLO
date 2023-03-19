from model import YOLOModel
from loss import YOLOLoss
from dataset import Dataset
from torch.utils.data import DataLoader
import torch

train_path = '../training_data/'

def train(epochs=8):
    loss = YOLOLoss(...)

    x = []
    y = []

    with open(train_path+'train/_annotations.csv', 'r') as f:
        line = f.readline()
        print(line)


    # Load dataset
    dataset = Dataset(x, y)

    # Create dataloader for dataset
    dataloader = DataLoader(dataset)

    # Calculate loss for batch
    preds = [YOLOModel(data) for data in batch]
    groud_truth = ...
    loss(preds, groud_truth)

if __name__ == "__main__":
    input("Press any key to start training the model...")
    train()
    print("Model successfully trained")