import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np


# ----- Loading the data and visualise it
# recommended CIFAR-10 mean/std (computed on CIFAR-10 training set)
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

# alter the dataset to create new sets
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# original dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# CIFAR-10 training set (50k images)
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transform_train, download=True
)

# CIFAR-10 test set (10k images)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform_test, download=True
)

# Load data
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# ----- Defining the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 3 channels -> RGB
        # First conv block: input 3 channels -> 32 feature maps, kernel 3x3, padding=1 keeps the same size of the original
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # batchnorm for the 32 channels

        # Second conv block: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pool : reduce spatial size with after each block (2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout : prevents overfitting
        self.dropout = nn.Dropout(p=0.25)

        # After two poolings: input 32x32 -> 16x16 (after 1st) -> 8x8 (after 2nd)
        # The number of channels after conv2 is 64 => flattened features = 64 * 8 * 8 = 4096
        # 512 neurons -> 10 classes
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # x shape: (batch_size, 3, 32, 32)
        x = self.conv1(x) # (batch_size, 32, 32, 32)
        x = self.bn1(x) # batch normalization
        x = F.relu(x)
        x = self.pool(x) # downsample -> (batch_size, 32, 16, 16)

        x = self.conv2(x) # (batch_size, 64, 16, 16)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x) # (batch_size, 64, 8, 8)

        x = self.dropout(x) # regularization (only active during training)

        x = x.view(x.size(0), -1)  # flatten -> (batch_size, 4096)

        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = self.dropout(x) # optional dropout in FC
        x = self.fc2(x) # (batch_size, 10)
        return x


# ----- Training setup
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(Device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ----- Training method
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ----- Evaluating method
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ----- Actually training
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, Device)

    print(f"Epoch {epoch}/{num_epochs} "
          f"- Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} "
          f"- Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

# Saving the model
torch.save(model.state_dict(), "cifar10_cnn.pth")
print("Model saved as cifar10_cnn.pth")


# ----- Actually evaluating
# Get one batch of test data
images, labels = next(iter(test_loader))
images = images.to(Device)
labels = labels.to(Device)

model.eval()
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Function to unnormalize image for display
def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))   # (channels, height, width) -> (height, width, channels)
    img = std * img + mean # unnormalize
    img = np.clip(img, 0, 1) # keeps values in [0,1]
    return img

# Plot 25 images in a 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    img = imshow(images[i])
    ax.imshow(img)
    true_label = classes[labels[i].item()]
    pred_label = classes[preds[i].item()]
    color = "green" if true_label == pred_label else "red"
    ax.set_title(f"T:{true_label}\nP:{pred_label}", color=color, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
