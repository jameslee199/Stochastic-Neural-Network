import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim

# -----------------------------
# Model Definition
# -----------------------------
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()
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

    def normalize_to_01(self, x):
        # Normalize each sample to [0,1] across all channels and spatial dimensions
        x_min = x.view(x.size(0), -1).min(dim=1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
        x_max = x.view(x.size(0), -1).max(dim=1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def normalize_fc(self, x):
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)

    def forward(self, x):
        # ----- First conv block -----
        x = self.conv1(x)
        x = torch.clamp(x, 0, 1)
        x = F.relu(x)

        x = self.conv2(x)
        x = torch.clamp(x, 0, 1)
        x = F.relu(x)
        x = self.pool1(x)

        # ----- Second conv block -----
        x = self.conv3(x)
        x = torch.clamp(x, 0, 1)
        x = F.relu(x)
        x = self.conv4(x)
        x = torch.clamp(x, 0, 1)
        x = F.relu(x)
        x = self.pool2(x)

        # ----- Fully connected layers -----
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Output logits
        return x


# -----------------------------
# Data Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False
)


# -----------------------------
# Training Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CIFAR10_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(50):  # train for 100 epochs
    running_loss = 0.0
    net.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}")

print("Training done!")


# -----------------------------
# Save Model
# -----------------------------
torch.save(net.state_dict(), "cifar10_cnn_unsign.pth")
print("Model saved as cifar10_cnn.pth ✅")


# -----------------------------
# Evaluation
# -----------------------------
net.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

