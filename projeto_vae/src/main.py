"""Ponto de entrada.

Uso (a partir de projeto_vae/):
  python database/database_install.py          # baixa o dataset (1x)
  python src/main.py search                    # busca de hiperparâmetros
  python src/main.py train                     # treino final
  python src/main.py analyze                   # gráficos e métricas
  python src/main.py all                       # tudo em sequência
  python src/main.py search --latent 2         # só uma dimensão latente
"""
import argparse

import config
from analysis import analyze
from search import run_search
from train import get_device, train_final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["search", "train", "analyze", "all"])
    p.add_argument("--latent", type=int, nargs="+", default=config.LATENT_DIMS)
    a = p.parse_args()

    device = get_device()
    print(f"Dispositivo: {device}")
    for d in a.latent:
        if a.cmd in ("search", "all"):
            run_search(d, device)
        if a.cmd in ("train", "all"):
            train_final(d, device)
        if a.cmd in ("analyze", "all"):
            analyze(d, device)


if __name__ == "__main__":
    main()
