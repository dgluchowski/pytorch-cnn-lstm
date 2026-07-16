"""MNIST digit classification with a small CNN (PyTorch).

Two convolutional blocks (32 -> 64 channels, each Conv -> ReLU -> MaxPool)
followed by a fully-connected head. Trains with Adam + cross-entropy, tracks
train/validation curves, and reports test accuracy.

Run:
    python image_classification/mnist_cnn.py

The MNIST dataset is downloaded automatically by torchvision into ../data
(git-ignored); nothing is committed to the repo.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
training_data = torchvision.datasets.MNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)
testing_data = torchvision.datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
)

# Split the training data into train and validation
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(correct / total)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(correct / total)

    print(
        f"Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, "
        f"Val Loss {val_losses[-1]:.4f}, Train Acc {train_accs[-1]:.4f}, "
        f"Val Acc {val_accs[-1]:.4f}"
    )

# Save learning curves
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss vs Epochs")
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accs, label="Train Acc")
plt.plot(range(1, epochs + 1), val_accs, label="Val Acc")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy vs Epochs")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mnist_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# Test accuracy
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f"Testing Accuracy: {100 * correct / total:.2f}%")
