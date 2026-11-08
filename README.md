# Generative Neural Network – Variational Autoencoder (VAE)

## Overview

This project implements and evaluates a **Variational Autoencoder (VAE)** using the Fashion-MNIST dataset.

The main objective is to explore generative neural networks, studying how they learn image representations, reconstruct input data, and generate new synthetic images.

## Dataset

The project uses [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), a dataset containing 70,000 grayscale images of clothing items.

- Image dimensions: 28 × 28 pixels
- 10 different classes
- 60,000 training images
- 10,000 test images

The training set was further divided into 55,000 training samples and 5,000 validation samples.

## Methodology

Two VAE configurations were implemented and compared:

- **Latent dimension = 2**
- **Latent dimension = 8**

The experiments included:

- Random hyperparameter search.
- Training with the Adam optimizer.
- Early stopping based on validation loss.
- Latent space visualization and analysis.
- Image reconstruction and generation.
- Clustering and latent space evaluation using PCA, K-Means, Silhouette Score, ARI and NMI.

## Results

The model with 8 latent dimensions achieved a lower test loss than the model with 2 dimensions.

| Metric | VAE (z=2) | VAE (z=8) |
|---|---:|---:|
| Validation Loss | 253.32 | 237.32 |
| Test Loss | 254.01 | 238.40 |
| Reconstruction Loss | 246.74 | 225.19 |
| KL Divergence | 7.27 | 13.21 |

The results suggest that the larger latent space provides better image reconstruction and more varied generated samples.

## Technologies and Concepts

- Python
- Artificial Neural Networks
- Variational Autoencoders (VAE)
- Fashion-MNIST
- Hyperparameter Optimization
- Principal Component Analysis (PCA)
- K-Means Clustering

## Repository Structure

```text
Projeto_Generativo/
├── projeto_vae/
├── Proj_RedeGenerativa_VAE.pdf
└── README.md
```

## Author

**Enzo Rosim**

Academic project developed for the course *Introduction to Artificial Neural Networks*.

[GitHub Repository](https://github.com/EnzoRosim/Projeto_Generativo)
