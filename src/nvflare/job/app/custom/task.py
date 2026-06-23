import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        try:
            self.model = models.resnet18(weights=None)
        except TypeError:
            self.model = models.resnet18(pretrained=False)

        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)


def _get_cifar10_root():
    return os.environ.get(
        "CIFAR10_DIR",
        os.path.expanduser("~/Escritorio/tfg/NFLARE/downloads"),
    )


def _cifar10_is_downloaded(root):
    return os.path.isdir(os.path.join(root, "cifar-10-batches-py"))


def _ensure_cifar10(root):
    os.makedirs(root, exist_ok=True)

    if not _cifar10_is_downloaded(root):
        print("[DATA] CIFAR-10 no encontrado. Descargando una vez...")
        datasets.CIFAR10(root=root, train=True, download=True)
        datasets.CIFAR10(root=root, train=False, download=True)
    else:
        print(f"[DATA] CIFAR-10 ya existe en: {root}")


def _get_partition(indices, partition_id, num_partitions):
    total = len(indices)
    part_size = total // num_partitions

    start = partition_id * part_size

    if partition_id == num_partitions - 1:
        end = total
    else:
        end = start + part_size

    return indices[start:end]


def load_data(partition_id, num_partitions, batch_size):
    root = _get_cifar10_root()
    _ensure_cifar10(root)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=train_transform,
    )

    val_dataset_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=val_transform,
    )

    generator = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(len(train_dataset_full), generator=generator).tolist()

    train_indices_all = all_indices[:40000]
    val_indices_all = all_indices[40000:]

    train_indices = _get_partition(
        train_indices_all,
        partition_id,
        num_partitions,
    )

    val_indices = _get_partition(
        val_indices_all,
        partition_id,
        num_partitions,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    use_cuda = torch.cuda.is_available()

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    return trainloader, valloader


def train(model, trainloader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    model.train()

    total_loss = 0.0
    total_steps = 0

    print("[TRAIN] starting training")
    print(f"[TRAIN] epochs={epochs}")
    print(f"[TRAIN] total_batches_per_epoch={len(trainloader)}")
    print(f"[TRAIN] device={device}")
    print(f"[TRAIN] torch_num_threads={torch.get_num_threads()}")

    for epoch in range(epochs):
        print(f"[TRAIN] epoch {epoch + 1}/{epochs} started")

        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if epoch == 0 and batch_idx == 0:
                print(f"[TRAIN BATCH] images device: {images.device}")
                print(f"[TRAIN BATCH] labels device: {labels.device}")
                print(f"[TRAIN BATCH] model device: {next(model.parameters()).device}")
                print(f"[TRAIN BATCH] images shape: {images.shape}")
                print(f"[TRAIN BATCH] labels shape: {labels.shape}")

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / total_steps
                print(
                    f"[TRAIN] epoch={epoch + 1}/{epochs} "
                    f"batch={batch_idx + 1}/{len(trainloader)} "
                    f"loss={loss.item():.4f} "
                    f"avg_loss={avg_loss:.4f}"
                )

        print(f"[TRAIN] epoch {epoch + 1}/{epochs} finished")

    avg_train_loss = total_loss / total_steps

    print(f"[TRAIN] finished. avg_trainloss={avg_train_loss:.4f}")

    return avg_train_loss


def test(model, testloader, device):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_steps = 0

    correct = 0
    total = 0

    print("[TEST] starting validation")

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_steps += 1

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total_steps
    accuracy = correct / total

    print(f"[TEST] val_loss={avg_loss:.4f}")
    print(f"[TEST] accuracy={accuracy:.4f}")

    return avg_loss, accuracy



def load_centralized_dataset(batch_size):
    root = _get_cifar10_root()
    _ensure_cifar10(root)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=False,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
