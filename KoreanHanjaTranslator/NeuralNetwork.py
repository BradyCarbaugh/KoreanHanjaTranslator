# TRAINING

import os
import torch
from torch import nn
import Images
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import ImageClassification
from DataSet import KoreanHanjaCharacterDataset

# Using cpu
device = "cpu"


# Defining Class
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


# Creating Model and moving it to device(cpu)
# model = NeuralNetwork().to(device)
# model = KoreanHanjaCharacterDataset().to(device)
model = KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', Images).to(device)
# print(model)

# Calling model on input and getting prediction probabilities
x = torch.rand(1, 28, 28, device=device)
logits = model
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicited Class: {y_pred}")

input_image = torch.rand(3, 28, 28)
print(input_image.size())

# Converting images into arrays
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# Applying a linear transformation on the input / weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# Non-linear activations
# Creating mappings between model's input and output for the neural network to learn
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLUE: {hidden1}\n\n")

seq_modeule = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modeule(input_image)

# Logits scaled to represent the model's predicted probabilities
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print(f"Model Structure: {model}\n\n")

# Model Parameters
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size} | Values: {param[:2]} \n")

