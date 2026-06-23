"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


class Net(nn.Module):
    """ResNet-18 adapted for CIFAR-10.

    The original ResNet-18 is designed for ImageNet images of size 224x224.
    CIFAR-10 images are 32x32, so we replace the first convolution and remove
    the initial maxpool layer.
    """

    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None, num_classes=10)

        # Adapt ResNet-18 for CIFAR-10: 32x32 RGB images
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove maxpool because CIFAR-10 images are small
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


# Cache FederatedDataset
fds = None


# Data augmentation for training
train_transforms = Compose(
    [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


# No augmentation for validation/test
test_transforms = Compose(
    [
        ToTensor(),
        Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


def apply_train_transforms(batch):
    """Apply training transforms to a batch."""
    batch["img"] = [train_transforms(img) for img in batch["img"]]
    return batch


def apply_test_transforms(batch):
    """Apply test transforms to a batch."""
    batch["img"] = [test_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load one CIFAR-10 partition and return train/test dataloaders."""

    global fds

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    # Load client partition
    partition = fds.load_partition(partition_id)

    # Split local data: 80% train, 20% validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_dataset = partition_train_test["train"].with_transform(
        apply_train_transforms
    )
    test_dataset = partition_train_test["test"].with_transform(
        apply_test_transforms
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return trainloader, testloader


def load_centralized_dataset():
    """Load the full CIFAR-10 test set and return a dataloader."""

    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    test_dataset = test_dataset.with_transform(apply_test_transforms)

    return DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def train(net, trainloader, epochs, lr, device):
    """Train the model on the local training set."""

    net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    net.train()

    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            if num_batches == 0:
                print(f"[TRAIN BATCH] images device: {images.device}")
                print(f"[TRAIN BATCH] labels device: {labels.device}")
                print(f"[TRAIN BATCH] model device: {next(net.parameters()).device}")
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

    avg_trainloss = running_loss / num_batches
    return avg_trainloss


def test(net, testloader, device):
    """Evaluate the model on a validation/test set."""

    net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    correct = 0
    total = 0
    loss = 0.0
    num_batches = 0

    net.eval()

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            batch_loss = criterion(outputs, labels)

            loss += batch_loss.item()
            num_batches += 1

            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = loss / num_batches

    return avg_loss, accuracy