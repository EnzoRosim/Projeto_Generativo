"""Espaço latente: gráficos, PCA, silhouette, reconstruções e amostras."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import config
from data import get_loaders
from model import VAE
from train import model_path

PLOTS = config.OUT_DIR / "plots"


def load_model(d, device):
    ck = torch.load(model_path(d), map_location=device)
    m = VAE(d, ck["hp"]["hidden"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval(), ck


@torch.no_grad()
def encode_test(model, device):
    _, _, dl = get_loaders(256)
    mus, ys = [], []
    for x, y in dl:
        mus.append(model.encode(x.to(device))[0].cpu()); ys.append(y)
    return torch.cat(mus).numpy(), torch.cat(ys).numpy()


def scatter(z2, y, title, xl, yl, path):
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(z2[:, 0], z2[:, 1], c=y, cmap="tab10", s=3, alpha=0.6)
    plt.legend(sc.legend_elements()[0], config.CLASSES, fontsize=8,
               bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.title(title); plt.xlabel(xl); plt.ylabel(yl)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


@torch.no_grad()
def plot_images(model, d, device):
    _, _, dl = get_loaders(16)
    x = next(iter(dl))[0][:10].to(device)
    rec = torch.sigmoid(model.decode(model.encode(x)[0])).view(-1, 28, 28).cpu()
    gen = torch.sigmoid(model.decode(torch.randn(10, d, device=device))
                        ).view(-1, 28, 28).cpu()
    fig, ax = plt.subplots(3, 10, figsize=(12, 4))
    for i in range(10):
        for r, img in enumerate([x[i, 0].cpu(), rec[i], gen[i]]):
            ax[r, i].imshow(img, cmap="gray"); ax[r, i].axis("off")
    for r, t in enumerate(["Original", "Reconstr.", "Amostra"]):
        ax[r, 0].set_title(t, fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(PLOTS / f"imagens_z{d}.png", dpi=150); plt.close()


def plot_history(ck, d):
    h = np.array(ck["hist"])
    plt.figure(figsize=(6, 4))
    plt.plot(h[:, 0], label="treino"); plt.plot(h[:, 1], label="validação")
    plt.xlabel("época"); plt.ylabel("custo (ELBO neg.)"); plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS / f"curva_z{d}.png", dpi=150); plt.close()


def analyze(d, device):
    model, ck = load_model(d, device)
    mu, y = encode_test(model, device)
    info = {"latent_dim": d, "hp": ck["hp"], "val": ck["val"], "test": ck["test"]}

    if d == 2:
        scatter(mu, y, "Espaço latente 2D", "z1", "z2", PLOTS / "latente_z2.png")
    else:
        pca = PCA(n_components=2).fit(mu)
        var = pca.explained_variance_ratio_
        scatter(pca.transform(mu), y,
                f"Latente 8D -> PCA 2D ({var.sum():.1%} da variância)",
                f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})", PLOTS / "latente_z8_pca.png")
        info["pca_var_explicada"] = var.tolist()
        info["pca_var_total"] = float(var.sum())

    # Separação dos rótulos (-1 a 1; maior = melhor separado)
    info["silhouette_rotulos"] = float(
        silhouette_score(mu, y, sample_size=5000, random_state=config.SEED))

    plot_images(model, d, device)
    plot_history(ck, d)
    json.dump(info, open(config.OUT_DIR / f"resumo_z{d}.json", "w"), indent=2)
    print(f"[análise] z={d}: {info}")
