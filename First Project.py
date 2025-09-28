import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.fx.experimental.partitioner_utils import Device
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# ----- Loading the data and visualise it

# transform images into tensors
transform = transforms.ToTensor()

# training dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

# test dataset
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

# load data in batches of 64
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# visualise a sample
images, labels = next(iter(train_loader))  # get one batch
print("Batch shape:", images.shape)
print("Labels:", labels[:10])


# ----- Defining the model

# building Neural Network Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # initialize parent class

        # define layers
        self.fc1 = nn.Linear(784, 128)  # input -> hidden
        self.fc2 = nn.Linear(128, 10)  # hidden -> output

    def forward(self, x):
        # flatten the image (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(-1, 28 * 28)

        # apply hidden layer + activation
        x = F.relu(self.fc1(x))

        # output layer (logits)
        x = self.fc2(x)
        return x


# ----- Training setup
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(Device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# ----- Training method
# Epoch = one complete pass through the entire training dataset
def train_epoch(model, dataloader, optimizer, criterion, device):
    # switch model into training mode
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # loop over batches
    for images, labels in dataloader:
        # move to device
        images = images.to(device)
        labels = labels.to(device)

        # clear old gradients
        optimizer.zero_grad()

        # forward pass = compute model predictions
        outputs = model(images)

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # track metrics
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ----- Evaluating method
def evaluate(model, dataloader, criterion, device):
    # switch model into evaluation mode
    # (this disables dropout and uses running stats for batchnorm)
    model.eval()

    running_loss = 0.0 # keep track of total loss
    correct = 0 # number of correct predictions
    total = 0 # total number of samples

    with torch.no_grad(): # Turn off gradient calculations (saves memory + speed)
        # Loop through the test dataset
        for images, labels in dataloader:
            # Move data to the same device as the model
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)

            # compute loss
            loss = criterion(outputs, labels)

            # add batch loss to the running total
            running_loss += loss.item() * images.size(0)

            # get predicted class (index of highest logit)
            preds = outputs.argmax(dim=1)

            # count how many predictions were correct
            correct += (preds == labels).sum().item()

            # count total number of samples seen
            total += labels.size(0)

    # average loss
    epoch_loss = running_loss / total

    # accuracy
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ----- Actually training
num_epochs = 5

for epoch in range(1, num_epochs + 1):
    # Train for one full pass over training data
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Device)

    # Evaluate on test data
    val_loss, val_acc = evaluate(model, test_loader, criterion, Device)

    # Print progress for this epoch
    print(f"Epoch {epoch}/{num_epochs} "
          f"- Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} "
          f"- Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved!")


# ----- Actually evaluating
images, labels = next(iter(test_loader))
images, labels = images.to(Device), labels.to(Device)

model.eval()
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

# show 25 images in a 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(12, 12))

for i, ax in enumerate(axes.flat):  # flatten axes to loop easily
    ax.imshow(images[i].cpu().squeeze(), cmap="gray")
    ax.set_title(f"P:{preds[i].item()} / T:{labels[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()
