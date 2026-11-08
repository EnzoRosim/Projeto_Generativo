"""Baixa o Fashion-MNIST para database/data/. Rodar uma vez."""
from pathlib import Path
from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parent / "data"

if __name__ == "__main__":
    datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    print(f"Dataset salvo em {DATA_DIR}")
