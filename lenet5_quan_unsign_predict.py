import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import argparse
import os
import numpy as np

# -----------------------------
# 1. Pooling & Conv Error Config
# -----------------------------
ADD_POOLING_ERROR = True
ADD_CONV_ERROR   = True
APC_SIZE = 25

MU_PATH = "./pooling_error/unsigned/pool_mu_unsigned.csv"
SIGMA_PATH = "./pooling_error/unsigned/pool_sigma_unsigned.csv"

mu_matrix = np.genfromtxt(MU_PATH, delimiter=',')
sigma_matrix = np.genfromtxt(SIGMA_PATH, delimiter=',')

if mu_matrix.shape != sigma_matrix.shape:
    raise ValueError("mu and sigma matrices must have the same shape")
n_bits_max, k_max = mu_matrix.shape

def apply_pooling_error(x: torch.Tensor, n_bits: int, k: int) -> torch.Tensor:
    if not ADD_POOLING_ERROR:
        return x
    mu = float(mu_matrix[n_bits-1, k-1])
    sigma = float(sigma_matrix[n_bits-1, k-1])
    noise = torch.empty_like(x).normal_(mean=mu, std=sigma)
    return x + noise

# -----------------------------
# 2. Load conv error matrices
# -----------------------------
MU_PATH = "./conv_error/avg_mu_unsigned.csv"
SIGMA_PATH = "./conv_error/avg_sigma_unsigned.csv"

mu_matrix = np.genfromtxt(MU_PATH, delimiter=',')
sigma_matrix = np.genfromtxt(SIGMA_PATH, delimiter=',')

if mu_matrix.shape != sigma_matrix.shape:
    raise ValueError("mu and sigma matrices must have the same shape")
n_bits_max, k_max = mu_matrix.shape

def apply_conv_error(x: torch.Tensor, n_bits: int, k: int) -> torch.Tensor:
    if not ADD_CONV_ERROR:
        return x
    mu = float(mu_matrix[n_bits-1, k-1])
    sigma = float(sigma_matrix[n_bits-1, k-1])
    noise = torch.empty_like(x).normal_(mean=mu, std=sigma)
    return x + noise

# -----------------------------
# 3. Define LeNet5 with n_bits & k
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def linear_scale(self, x):
        max1 = math.log(APC_SIZE) + 1
        min1 = 0.0
        return torch.clamp((x - min1) / (max1 - min1), 0.0, 1.0)

    def apply_activation_quant(self, x, n_bits):
        if n_bits is None:
            return x
        step = 1 / (2**n_bits - 1)
        return torch.round(x / step) * step

    def forward(self, x, n_bits=None, k=None):
        # Conv1
        x = self.conv1(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, n_bits)
        x = apply_conv_error(x, n_bits, k)
        x = F.relu(x)
        x = self.pool1(x)
        x = apply_pooling_error(x, n_bits, k)

        # Conv2
        x = self.conv2(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, n_bits)
        x = apply_conv_error(x, n_bits, k)
        x = F.relu(x)
        x = self.pool2(x)
        x = apply_pooling_error(x, n_bits, k)

        # Fully connected
        x = x.view(-1, 16*4*4)

        x = self.fc1(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, n_bits)
        x = apply_conv_error(x, n_bits, k)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, n_bits)
        x = apply_conv_error(x, n_bits, k)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.apply_activation_quant(x, n_bits)
        x = apply_conv_error(x, n_bits, k)
        return x

# -----------------------------
# 4. Device setup & load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LeNet5().to(device)
model.load_state_dict(torch.load("lenet5_mnist_quan_unsign.pth", map_location=device))
model.eval()

def clip_parameters(model, min_val=0, max_val=1.0):
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(min_val, max_val)

clip_parameters(model, 0, 1)

# -----------------------------
# 5. Load test dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# -----------------------------
# 6. Prediction interface
# -----------------------------
def predict(model, test_loader, n_bits=None, k=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, n_bits=n_bits, k=k)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Prediction with n_bits={n_bits}, k={k}: Accuracy={acc:.2f}%")
    return acc

# # Example usage:
results = {}  # store results for each k if needed
ADD_POOLING_ERROR = True
ADD_CONV_ERROR   = True

for k in range(1, 96):   # k = 1 to 256
    results[k] = predict(model, test_loader, n_bits=6, k=k)

# optional: print them out
for k, res in results.items():
    print(f"k={k}: {res}")

# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Sweep k for given n_bits")
#     parser.add_argument("--n_bits", type=int, required=True, help="Number of bits for quantization")
#     args = parser.parse_args()
#
#     results = {}
#     for k in range(1, 257):  # sweep k from 1 to 256
#         acc = predict(model, test_loader, n_bits=args.n_bits, k=k)
#         results[k] = acc
