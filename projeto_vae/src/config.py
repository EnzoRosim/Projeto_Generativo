"""Configuração central: edite aqui os hiperparâmetros."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "database" / "data"
OUT_DIR = ROOT / "output"

SEED = 42
VAL_SIZE = 5000          # amostras do treino reservadas p/ validação
LATENT_DIMS = [2, 8]

CLASSES = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]

# ---------- Treino final (python main.py train) ----------
USE_BEST_FROM_SEARCH = True   # True: usa o melhor da busca; False: usa FINAL abaixo
FINAL_EPOCHS = 50
PATIENCE = 8                  # early stopping (épocas sem melhora na validação)
FINAL = {                     # hiperparâmetros manuais, por dim. latente
    2: dict(hidden=[512, 256], lr=1e-3, batch_size=128),
    8: dict(hidden=[512, 256], lr=1e-3, batch_size=128),
}

# ---------- Busca de hiperparâmetros (python main.py search) ----------
SEARCH_EPOCHS = 15
N_TRIALS = 8                  # nº de combinações sorteadas do grid
SEARCH = {
    "hidden": [[256], [512, 256], [512, 256, 128]],   # topologia
    "lr": [1e-3, 5e-4],
    "batch_size": [128, 256],
}
