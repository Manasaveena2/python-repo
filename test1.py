import warnings
warnings.filterwarnings("ignore")

# Python
import random
import os

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# Torchvison
from torchvision.utils import make_grid
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder

# Utils
import matplotlib.pyplot as plt

# Custom Loss: Discrepancy Loss
class DiscrepancyLoss(nn.Module):
    def __init__(self):
        super(DiscrepancyLoss, self).__init__()

    def forward(self, output1, output2):
        return torch.mean(torch.abs(output1.softmax(dim=1) - output2.softmax(dim=1)))

# Visualization Helper
def imshow(inp, title=None):
    """Display tensor images."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Define backbone model
class DenseNetTwoHead(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetTwoHead, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier1 = nn.Linear(64 * 16 * 16, num_classes)
        self.classifier2 = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out1 = self.classifier1(features)
        out2 = self.classifier2(features)
        return out1, out2

# Data transforms
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# Load CIFAR-10 (In-Distribution) dataset
cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
cifar10_val = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Load TinyImageNet (Out-of-Distribution) dataset
tinyimagenet_dir = "./data/tiny-imagenet-200/train"  # Verify this path!

# Download the TinyImageNet dataset if it doesn't exist
if not os.path.exists(tinyimagenet_dir):
    !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P ./data
    !unzip ./data/tiny-imagenet-200.zip -d ./data
    print("TinyImageNet dataset downloaded and extracted successfully.")
else:
    print("TinyImageNet dataset already exists.")

tinyimagenet_dataset = ImageFolder(root=tinyimagenet_dir, transform=test_transform)


# Dataloaders
batch_size = 64
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=False, num_workers=2)
ood_loader = DataLoader(tinyimagenet_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNetTwoHead(num_classes=10).to(device)
sup_criterion = nn.CrossEntropyLoss()
unsup_criterion = DiscrepancyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

# Training loop
def train_model(model, train_loader, optimizer, sup_criterion, unsup_criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out1, out2 = model(images)
            sup_loss = sup_criterion(out1, labels) + sup_criterion(out2, labels)
            disc_loss = unsup_criterion(out1, out2)
            loss = sup_loss - 0.1 * disc_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
def evaluate_model(model, id_loader, ood_loader):
    model.eval()
    id_discrepancies, ood_discrepancies = [], []
    with torch.no_grad():
        for images, _ in id_loader:
            images = images.to(device)
            out1, out2 = model(images)
            disc = unsup_criterion(out1, out2).item()
            id_discrepancies.append(disc)
        for images, _ in ood_loader:
            images = images.to(device)
            out1, out2 = model(images)
            disc = unsup_criterion(out1, out2).item()
            ood_discrepancies.append(disc)
    return id_discrepancies, ood_discrepancies

# Train the model
train_model(model, train_loader, optimizer, sup_criterion, unsup_criterion, epochs=10)

# Evaluate the model
id_discrepancies, ood_discrepancies = evaluate_model(model, val_loader, ood_loader)

# Example data (replace with your actual scores)
# Detection scores for CIFAR-100 (ID) and Tiny ImageNet-resized (OOD)
import matplotlib.pyplot as plt
import numpy as np
id_scores = np.random.normal(1.0, 0.1, 5000)  # Example ID scores
ood_scores = np.random.normal(0.2, 0.1, 5000)  # Example OOD scores

# Histogram for proposed method (our detector)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(id_scores, bins=50, alpha=0.7, label="CIFAR-100", color="blue")
plt.hist(ood_scores, bins=50, alpha=0.7, label="TINr", color="orange")
plt.title("Our Detector")
plt.xlabel("Detection Score")
plt.ylabel("Frequency")
plt.legend()

# Example scores for ELOC detector (replace with actual ELOC results)
id_scores_eloc = np.random.normal(-0.0065, 0.0001, 5000)
ood_scores_eloc = np.random.normal(-0.0068, 0.0001, 5000)

# Histogram for ELOC detector
plt.subplot(1, 2, 2)
plt.hist(id_scores_eloc, bins=50, alpha=0.7, label="CIFAR-100", color="blue")
plt.hist(ood_scores_eloc, bins=50, alpha=0.7, label="TINr", color="orange")
plt.title("ELOC Detector")
plt.xlabel("Detection Score")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

