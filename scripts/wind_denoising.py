""" Script for wind experiment in the journal paper """

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
from omegaconf import DictConfig
import seaborn as sns
import matplotlib.pyplot as plt

from src.builder import *
from src.solver import *
from src.utils.dsp_utils import *

@hydra.main(
    config_path="../configs",  
    config_name="wind_denoising",
    version_base="1.3"
)

def main(cfg: DictConfig):

    SNR : float = cfg.signals.SNR
    sparsity : int = cfg.signals.sparsity 
    lambda_ : float = cfg.signals.lambda_
    solver : str = cfg.solver

    # Read data 
    X = np.load('data/important/PeakWeather/wind_data_uv_full.npy')

    train_split_idx = int(0.8 * X.shape[1])
    X_test = X[:, train_split_idx:]

    X_noisy = add_noise_snr(X_test, snr_db = SNR)
    uuid : str = f'wind_{solver}.npy'
    LOAD_DIR : Path = Path('.') / 'data/interim/wind/' / uuid
    L_hat = np.load(LOAD_DIR)

    _, U = np.linalg.eigh(L_hat)
    H = np.linalg.inv(np.eye(L_hat.shape[0]) + lambda_ * L_hat)
    # print(nonlinear_approximation(
    #     U = U,
    #     X = X_noisy,
    #     M = sparsity,
    #     X_true = X_test,
    #     return_db=True
    # ))
    print(f'Error for {solver} at SNR={SNR}db, lambda={lambda_}:{10 * np.log10(np.linalg.norm(H @ X_noisy - X_test) ** 2 / np.linalg.norm(X_test) ** 2):.2f}')

if __name__ == '__main__':
    main()