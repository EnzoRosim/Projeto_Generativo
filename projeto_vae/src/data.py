"""Carregamento do Fashion-MNIST (treino / validação / teste)."""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def get_loaders(batch_size):
    tf = transforms.ToTensor()  # pixels em [0, 1]
    full = datasets.FashionMNIST(config.DATA_DIR, train=True, transform=tf)
    test = datasets.FashionMNIST(config.DATA_DIR, train=False, transform=tf)
    g = torch.Generator().manual_seed(config.SEED)
    train, val = random_split(
        full, [len(full) - config.VAL_SIZE, config.VAL_SIZE], generator=g)
    return (DataLoader(train, batch_size, shuffle=True),
            DataLoader(val, batch_size),
            DataLoader(test, batch_size))
