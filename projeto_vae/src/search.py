"""Busca aleatória de hiperparâmetros (validação)."""
import itertools
import json
import random

import config
from train import fit, best_path


def run_search(latent_dim, device):
    keys = list(config.SEARCH)
    grid = [dict(zip(keys, v)) for v in itertools.product(*config.SEARCH.values())]
    random.Random(config.SEED).shuffle(grid)
    trials = grid[:config.N_TRIALS]

    results = []
    for i, hp in enumerate(trials, 1):
        _, _, val = fit(latent_dim, hp, config.SEARCH_EPOCHS, device, verbose=False)
        results.append({**hp, "val_loss": val})
        print(f"[busca z={latent_dim}] {i}/{len(trials)} {hp} -> val {val:.2f}")

    results.sort(key=lambda r: r["val_loss"])
    out = config.OUT_DIR / "search"
    json.dump(results, open(out / f"search_z{latent_dim}.json", "w"), indent=2)
    best = {k: results[0][k] for k in keys}
    json.dump(best, open(best_path(latent_dim), "w"), indent=2)
    print(f"[busca z={latent_dim}] melhor: {best}")
