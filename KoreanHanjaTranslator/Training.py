# TRAINING

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from csv import reader
from PIL import Image
import DataSet
import Images
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

device = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)



class KoreanHanjaCharacterDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # Len of Dataset
        return len(self.img_labels)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Creating Datasets for Training and Validation
training_set = KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', Images)
validation_set = KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', Images)

# Creating Data Loaders for the datasets
training_loader = DataSet.KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', Images, transform = transform)
validation_loader = DataSet.KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', Images, transform = transform)

# Class Labels
classes = ('Strength', 'Peace', 'Mysterious', 'Amuse', 'Lightning', 'Learning')

print('Training Set has {} instances' .format(len(training_set)))
print('Validation Set has {} instances' .format(len(validation_set)))


# Helper function for inline image display --- Visualizing the Data
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    nping = img.numpy()
    if one_channel:
        plt.imshow(nping, cmap='Greys')
    else:
        plt.imshow(np.transpose(nping, (1, 2, 0)))


with open('HanjaCharacters.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    dataiter = csv_reader.__next__()
    for row in csv_reader:
        dataiter = row
dataiter = iter(training_loader)
images, labels = dataiter.__next__()

# Create a grid from the images
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print(classes[labels[j]] for j in range(4))

# Model
model = DataSet.KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', Images)

# Display image and label.
train_features, train_labels = next(iter(training_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {6}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()
dummy_outputs = torch.rand(4, 10)
dummy_labels = torch.tensor([1, 5, 3, 7])
loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))
# Optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Training Loop
# Gets batch of training data
# Calculates loss
# Reports per-batch loss
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        outputs = model.img_labels
        loss = loss_fn(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        # Gather Data and report
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

# Per-epoch Activity
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/Korean_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

# Preform validation by checking relative loss on the set of data
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model.img_labels

    avg_vloss = running_vloss / (epoch+1)

    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)

    epoch_number += 1

X = torch.rand(1, 28, 28, device=device)
logits = DataSet.KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', "Test_Pres.png")

input_image = torch.rand(3,28,28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits_1 = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits_1)
print("pred_probab: {}".format(pred_probab))
print("softmax: {}".format(softmax))
print("logits: {}".format(logits_1))

print(f"Model structure: {model}\n\n")


# Image Testing
image = Image.open("Test_Pres.png")
image.show()
p = DataSet.KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', image).img_labels
p_class = np.argmax(p)
p_guess = DataSet.KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', image).__getitem__(p_class)
print('Translation of Hanja Character: {}'.format(p_guess))