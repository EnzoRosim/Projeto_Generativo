"""Treino (fit), treino final e utilidades."""
import copy
import json
import random

import numpy as np
import torch

import config
from data import get_loaders
from model import VAE, vae_loss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=config.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, loader, device, opt=None):
    """Uma época; treina se opt for dado. Retorna custo médio por amostra."""
    training = opt is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            loss = vae_loss(logits, x, mu, logvar)
            if training:
                opt.zero_grad()
                (loss / x.size(0)).backward()
                opt.step()
            total += loss.item()
    return total / len(loader.dataset)


def fit(latent_dim, hp, epochs, device, patience=config.PATIENCE, verbose=True):
    """Treina com early stopping; retorna (modelo, histórico, melhor val)."""
    set_seed()
    train_dl, val_dl, _ = get_loaders(hp["batch_size"])
    model = VAE(latent_dim, hp["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    best, best_state, wait, hist = float("inf"), None, 0, []
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_dl, device, opt)
        va = run_epoch(model, val_dl, device)
        hist.append((tr, va))
        if verbose:
            print(f"  ép {ep:3d} | treino {tr:.2f} | val {va:.2f}")
        if va < best:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist, best


def best_path(d):
    return config.OUT_DIR / "search" / f"best_z{d}.json"


def model_path(d):
    return config.OUT_DIR / "models" / f"vae_z{d}.pt"


def train_final(latent_dim, device):
    """Treina o modelo final (hiperparâmetros da busca ou do config.FINAL)."""
    if config.USE_BEST_FROM_SEARCH and best_path(latent_dim).exists():
        hp = json.load(open(best_path(latent_dim)))
    else:
        hp = config.FINAL[latent_dim]
    print(f"[treino] z={latent_dim} hp={hp}")
    model, hist, best = fit(latent_dim, hp, config.FINAL_EPOCHS, device)

    _, _, test_dl = get_loaders(hp["batch_size"])
    test = run_epoch(model, test_dl, device)
    print(f"[treino] z={latent_dim} val={best:.2f} teste={test:.2f}")

    torch.save({"state": model.state_dict(), "latent_dim": latent_dim,
                "hp": hp, "hist": hist, "val": best, "test": test},
               model_path(latent_dim))
