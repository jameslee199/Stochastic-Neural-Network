import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# 1. Pooling & Conv Error Config
# -----------------------------
AVG_RESULTS_PATH = "avg_results.csv"  # header: avg_mu, avg_sigma
ADD_POOLING_ERROR = False  # Set False to disable pooling error
ADD_CONV_ERROR   = False  # Set False to disable conv error

_pooling_tables_cache = {}  # cache tables for different n_bits

def load_pooling_table(n_bits):
    if n_bits in _pooling_tables_cache:
        return _pooling_tables_cache[n_bits]
    pooling_table_path = f"pool_{n_bits}.csv"
    data = np.genfromtxt(pooling_table_path, delimiter=',', skip_header=1)
    out_real = data[:, 0]
    out_sc   = data[:, 1]
    error    = data[:, 2]
    # Precompute mapping: sc_pool_val -> array of candidate errors
    candidates_dict = defaultdict(list)
    for sc, err in zip(out_sc, error):
        candidates_dict[sc].append(err)
    candidates_dict = {k: np.array(v) for k, v in candidates_dict.items()}
    _pooling_tables_cache[n_bits] = (out_real, out_sc, error, candidates_dict)
    return _pooling_tables_cache[n_bits]

def generate_pooling_error_table(out_values, n_bits: int):
    _, out_sc, _, candidates_dict = load_pooling_table(n_bits)
    out_values_np = out_values.detach().cpu().numpy().ravel()

    # Sort out_sc for searchsorted
    sorted_idx = np.argsort(out_sc)
    out_sc_sorted = out_sc[sorted_idx]

    # Find nearest indices
    idx = np.searchsorted(out_sc_sorted, out_values_np, side='left')
    idx = np.clip(idx, 1, len(out_sc_sorted)-1)
    left = out_sc_sorted[idx-1]
    right = out_sc_sorted[idx]
    idx_nearest = np.where(np.abs(out_values_np - left) <= np.abs(out_values_np - right),
                           idx-1, idx)
    nearest_vals = out_sc_sorted[idx_nearest]

    # Vectorized random selection of candidates
    rand_choices = np.zeros_like(out_values_np)
    for val in np.unique(nearest_vals):
        mask = nearest_vals == val
        candidates = candidates_dict[val]
        rand_choices[mask] = np.random.choice(candidates, size=mask.sum())

    return torch.tensor(rand_choices.reshape(out_values.shape),
                        dtype=out_values.dtype, device=out_values.device)

def vectorized_pooling_error(pooled, n_bits: int):
    flat = pooled.view(pooled.size(0), pooled.size(1), -1)
    e_tensor = generate_pooling_error_table(flat, n_bits)
    return pooled + e_tensor.view_as(pooled)

def apply_pooling_error(pooled, n_bits):
    if ADD_POOLING_ERROR:
        return vectorized_pooling_error(pooled, n_bits)
    else:
        return pooled

def apply_conv_error(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    if not ADD_CONV_ERROR:
        return x
    if n_bits - 2 >= len(conv_error_df):
        raise ValueError(f"n_bits={n_bits} exceeds avg_results.csv rows={len(conv_error_df)}")
    row_index = n_bits - 2
    row = conv_error_df.iloc[row_index]
    mu = float(row['avg_mu'])
    sigma = float(row['avg_sigma'])
    noise = torch.empty_like(x).normal_(mean=mu, std=sigma)
    return x + noise

# -----------------------------
# 2. Load conv error CSV
# -----------------------------
conv_error_df = pd.read_csv(AVG_RESULTS_PATH)

# -----------------------------
# 3. Activation quantization
# -----------------------------
def apply_activation_quant(x, n_bits):
    if n_bits is None:   # skip quantization
        return x
    step = 2 / (2**n_bits - 1)  # [-1,1] range
    return torch.round(x / step) * step

# -----------------------------
# 4. CIFAR-10 CNN definition
# -----------------------------
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10, n_bits=8):
        super(CIFAR10_CNN, self).__init__()
        self.n_bits = n_bits
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)  # -> 30x30
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0) # -> 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # -> 14x14

        # Second conv block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # -> 12x12
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0) # -> 10x10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # -> 5x5

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = torch.clamp(x, -1, 1)
        x = apply_activation_quant(x, self.n_bits)
        x = apply_conv_error(x, self.n_bits)
        x = F.relu(x)

        # Conv2 + Pool
        x = self.conv2(x)
        x = torch.clamp(x, -1, 1)
        x = apply_activation_quant(x, self.n_bits)
        x = apply_conv_error(x, self.n_bits)
        x = F.relu(x)
        x = self.pool1(x)
        x = apply_pooling_error(x, self.n_bits)

        # Conv3
        x = self.conv3(x)
        x = torch.clamp(x, -1, 1)
        x = apply_activation_quant(x, self.n_bits)
        x = apply_conv_error(x, self.n_bits)
        x = F.relu(x)

        # Conv4 + Pool
        x = self.conv4(x)
        x = torch.clamp(x, -1, 1)
        x = apply_activation_quant(x, self.n_bits)
        x = apply_conv_error(x, self.n_bits)
        x = F.relu(x)
        x = self.pool2(x)
        x = apply_pooling_error(x, self.n_bits)

        # FC layers
        x = x.view(x.size(0), -1)  # now (64*5*5) = 1600
        x = self.fc1(x)
        x = torch.clamp(x, -1, 1)
        x = apply_activation_quant(x, self.n_bits)
        x = apply_conv_error(x, self.n_bits)
        x = F.relu(x)

        x = self.fc2(x)
        x = torch.clamp(x, -1, 1)
        x = apply_activation_quant(x, self.n_bits)
        x = apply_conv_error(x, self.n_bits)
        return x


# -----------------------------
# 5. Quantize model parameters
# -----------------------------
def quantize_model_params(model):
    n_bits = model.n_bits
    step = 2 / (2**n_bits - 1)
    with torch.no_grad():
        for p in model.parameters():
            p.clamp_(-1, 1)
            p.copy_(torch.round(p/step) * step)

# -----------------------------
# 6. Load base model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = CIFAR10_CNN().to(device)

try:
    state_dict = torch.load("cifar10_cnn.pth", map_location=device)
    base_model.load_state_dict(state_dict)
    print("Loaded pretrained model.")
except FileNotFoundError:
    print("No pretrained model found, using random initialization.")

# -----------------------------
# 7. CIFAR-10 test data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
try:
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
except:
    from torchvision.datasets import FakeData
    print("CIFAR-10 download failed, using FakeData.")
    test_dataset = FakeData(size=1000, image_size=(3,32,32), num_classes=10, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 8. Evaluate function
# -----------------------------
def evaluate_model(n_bits):
    model = CIFAR10_CNN(n_bits=n_bits).to(device)
    model.load_state_dict(base_model.state_dict())
    quantize_model_params(model)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# -----------------------------
# 9. Sweep n_bits and plot
# -----------------------------
bits_list = list(range(2, 9))  # n_bits = 2..8
accuracy_with_error = []
accuracy_without_error = []

for n_bits in bits_list:
    # With errors
    ADD_POOLING_ERROR = True
    ADD_CONV_ERROR = True
    acc_err = evaluate_model(n_bits)
    accuracy_with_error.append(acc_err)

    # Without errors
    ADD_POOLING_ERROR = False
    ADD_CONV_ERROR = False
    acc_noerr = evaluate_model(n_bits)
    accuracy_without_error.append(acc_noerr)

    print(f"n_bits={n_bits}, With Error={acc_err:.2f}%, Without Error={acc_noerr:.2f}%")

plt.figure(figsize=(8,5))
plt.plot(bits_list, accuracy_with_error, marker="o", label="With Error")
plt.plot(bits_list, accuracy_without_error, marker="s", label="Without Error")
plt.xlabel("Quantization bits (n_bits)")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Quantization Bits for CIFAR-10 CNN")
plt.ylim([0, 100])
plt.grid(True)
plt.legend()
plt.show()
