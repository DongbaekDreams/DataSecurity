'''Single course-aligned MNIST loader for Levels 1, 2, and 3.

Matches CSC8370_26FullDemo templates:
  - root: ./data/mnist
  - transform: ToTensor + Normalize((0.5,), (0.5,))
  - Level 1: train batch 50, test batch = full test set
  - Levels 2/3: equal 10-way partition, client batch 50, test batch 1000
'''

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

MNIST_ROOT = "./data/mnist"
CLIENT_BATCH = 50
TEST_BATCH_FED = 1000


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def load_mnist():
    '''Raw train/test datasets (same for every level).'''
    transform = mnist_transform()
    train_dataset = torchvision.datasets.MNIST(
        root=MNIST_ROOT, train=True, download=True, transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=MNIST_ROOT, train=False, download=True, transform=transform,
    )
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return train_dataset, test_dataset


def load_central(batch_size=CLIENT_BATCH):
    '''Level 1: centralized train/test loaders (course dataloader4level1 pattern).'''
    train_dataset, test_dataset = load_mnist()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    print(f"training has: {len(train_loader)} batch of data!")
    print(f"testing has: {len(test_loader)} batch of data!")
    return train_loader, test_loader


def partition_dataset(dataset, n_clients=10):
    '''Equal-size partitions; no example shared across clients (course Level 2 pattern).'''
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)


def load_federated(n_clients=10, client_batch=CLIENT_BATCH, test_batch=TEST_BATCH_FED):
    '''Levels 2 and 3: same data, same partitions, same loaders.'''
    train_dataset, test_dataset = load_mnist()
    client_datasets = partition_dataset(train_dataset, n_clients=n_clients)
    client_loaders = [
        DataLoader(ds, batch_size=client_batch, shuffle=True) for ds in client_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)
    for i, ds in enumerate(client_datasets):
        print(f"  client {i}: {len(ds)} examples")
    return client_datasets, client_loaders, test_loader
