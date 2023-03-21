from model import YOLOModel
from utils.loss import YOLOLoss
from dataset import Dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

train_path = '../training_data/'

def get_dataset(path="train"):
    x = []
    y = []

    last_img = ''
    # Load classes
    classes = dict()
    with open(f'{train_path}/{path}/_classes.txt', 'r') as f:
        lines = f.read().split('\n')
        classes = {line:i+1 for i, line in enumerate(lines)}
    class_count = len(classes)

    img_tensor = transforms.ToTensor()

    # Load data
    with open(f'{train_path}/{path}/_annotations.csv', 'r') as f:
        f.readline()
        while ',' in (line:=f.readline()):
            data = line[:-1].split(',')
            data[3] = classes[data[3]]
            # format to: [[pc, bx, by, bh, bw, c1, c2,..., cn],]
            data_x = [float(x) for x in data[1:]]
            data_x[2] = int(data_x[2])
            # calculate rescale factor
            res = (448/data_x[0], 448/data_x[1])
            x_ = [1, ((data_x[3]+data_x[5])/2)*res[0], ((data_x[4]+data_x[6])/2)*res[0], data_x[1]**res[0], data_x[0]*res[0]] + (data_x[2]-1)*[0] + [1] + (class_count-data_x[2])*[0]
            if data[0] == last_img:
                x[-1].append(x_)
            else:
                last_img = data[0]
                x.append([x_])

                # Load image
                img = Image.open(f'{train_path}/{path}/'+data[0])
                img = img.resize((448, 448))
                y.append(img_tensor(img))

    return Dataset(x, y, class_count=class_count)

def train(epochs=8):

    # Load dataset
    dataset = get_dataset()

    # Create dataloader for dataset
#    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model
    global model
    model = YOLOModel('struct.nn')

    # Initialize loss function
    global loss
    loss = YOLOLoss(n_classes=dataset.class_count, n_grid_cells=7)

    #for batch in iter(dataloader):
    batch = dataset[:20]
    # Calculate loss for batch
    preds = []
    for data in batch[0]:
        print(f'DATA:\n{data}\n\n')
        preds.append(model(data[0]))
    print(f"Predictions: \n{preds}\n")
    groud_truth = [data[1] for data in batch]
    print(f"Ground truth: \n{groud_truth}\n")
    l = loss(preds, groud_truth)


def validate():

    dataset = get_dataset(path="valid")

    preds = [YOLOModel(data[0]) for data in dataset]
    ground_truth = [data[1] for data in dataset]
    return loss(preds, ground_truth)



if __name__ == "__main__":
    #input("Press any key to start training the model...")
    print("Training Model")
    train()
    print("Model successfully trained")
    input("Press any key to start validating the model...")
    loss = validate()
    print(f"Model successfully trained. Loss: {loss}")
