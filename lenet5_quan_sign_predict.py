import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# -----------------------------
# 1. Conv error config
# -----------------------------
ADD_CONV_ERROR = True
APC_SIZE = 25

MU_PATH = "./pooling_error/signed/pool_mu_signed.csv"
SIGMA_PATH = "./pooling_error/signed/pool_sigma_signed.csv"

mu_matrix = np.genfromtxt(MU_PATH, delimiter=',')
sigma_matrix = np.genfromtxt(SIGMA_PATH, delimiter=',')

if mu_matrix.shape != sigma_matrix.shape:
    raise ValueError("mu and sigma matrices must have the same shape")

def apply_conv_error(x: torch.Tensor, n_bits: int, k: int) -> torch.Tensor:
    if not ADD_CONV_ERROR:
        return x
    mu = float(mu_matrix[n_bits-1, k-1])
    sigma = float(sigma_matrix[n_bits-1, k-1])
    noise = torch.empty_like(x).normal_(mean=mu, std=sigma)
    return x + noise

# -----------------------------
# 2. Define LeNet-5 model
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
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
        step = 2 / (2**n_bits - 1)
        return torch.round(x / step) * step

    def forward(self, x):
        x = self.conv1(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, 8)
        x = apply_conv_error(x, 8, 256)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, 8)
        x = apply_conv_error(x, 8, 256)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, 8)
        x = apply_conv_error(x, 8, 256)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.linear_scale(x)
        x = self.apply_activation_quant(x, 8)
        x = apply_conv_error(x, 8, 256)
        x = F.relu(x)

        x = self.fc3(x)
        return x

# -----------------------------
# 3. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 4. Load dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

# -----------------------------
# 5. Load model
# -----------------------------
model = LeNet5().to(device)
model.load_state_dict(torch.load("lenet5_mnist_quan_sign.pth", map_location=device))
model.eval()

# -----------------------------
# 6. Prediction & accuracy
# -----------------------------
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"MNIST Test Accuracy: {100 * correct / total:.2f}%")
