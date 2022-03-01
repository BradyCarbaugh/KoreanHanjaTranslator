# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np

print(torch.__version__)

data = [[1, 2], [3, 4]]
print(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

tensor = torch.rand(3, 4)
print(f"Tensor Shape: \n {tensor.shape} \n")
print(f"Tensor Datatype: \n {tensor.dtype} \n")
print(f"Tensor Device: \n {tensor.device} \n")

x = 5
if x == 5:
    print(x)

