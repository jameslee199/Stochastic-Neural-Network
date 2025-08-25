import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import math

print("Torch version:", torch.__version__)

ADD_CONV_ERROR   = True

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
# 1. Define the LeNet-5 model
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # Input: 1 channel (MNIST)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjusted for 28x28 input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # 10 classes (digits 0–9)

    def linear_scale(self, x):
        max1 = math.log(25) + 1  # ≈ 3.2189
        min1 = 0.0
        return torch.clamp((x - min1) / (max1 - min1), 0.0, 1.0)

    # -------------------------
    # Weight quantization
    # -------------------------
    def linear_scale_weight(self, w):
        """Scale weight to [0,1]"""
        w_min = w.min()
        w_max = w.max()
        return (w - w_min) / (w_max - w_min + 1e-8)

    def apply_weight_quant(self, w, n_bits):
        """Quantize weight tensor to n_bits"""
        if n_bits is None:
            return w
        w_scaled = self.linear_scale_weight(w)
        step = 1 / (2 ** n_bits - 1)
        return torch.round(w_scaled / step) * step

    def quantize_all_weights(self, n_bits):
        """Quantize all weights and biases in the model"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(self.apply_weight_quant(param, n_bits))

    def forward(self, x):
        # First conv + clamp + relu + pool
        x = self.conv1(x)
        x = self.linear_scale(x)
        x = apply_conv_error(x, 4, 16)
        x = F.relu(x)
        x = self.pool1(x)

        # Second conv + clamp + relu + pool
        x = self.conv2(x)
        x = self.linear_scale(x)
        x = apply_conv_error(x, 4, 16)
        x = F.relu(x)
        x = self.pool2(x)

        # Fully connected layers
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.linear_scale(x)
        x = apply_conv_error(x, 4, 16)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.linear_scale(x)
        x = apply_conv_error(x, 4, 16)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.relu(x)
        return x


# -----------------------------
# 2. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# 3. Data preparation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

# -----------------------------
# 4. Model, loss, optimizer
# -----------------------------
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def clip_parameters(model, min_val=0, max_val=1.0):
    """Clamp all weights and biases in-place."""
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(min_val, max_val)

def progress_bar(batch_idx, total_batches, bar_length=30):
    progress = batch_idx / total_batches
    filled_len = int(bar_length * progress)
    bar = "=" * filled_len + "-" * (bar_length - filled_len)
    percent = int(100 * progress)
    sys.stdout.write(f"\r[{bar}] {percent}%")
    sys.stdout.flush()

# ----------------------------
# Training loop with progress
# ----------------------------
for epoch in range(100):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        clip_parameters(model, 0, 1)

        running_loss += loss.item()

        # 🔹 Show progress bar
        progress_bar(batch_idx, total_batches)

    print(f"\nEpoch [{epoch+1}/20], Loss: {running_loss/total_batches:.4f}")


# Save once at the end
torch.save(model.state_dict(), "lenet5_mnist_quan_unsign.pth")
print("Model saved to lenet5_mnist_quan_unsign.pth")


# -----------------------------
# 6. Evaluation
# -----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
