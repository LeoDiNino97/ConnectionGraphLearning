""" Script for wind experiment in the journal paper """

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
from omegaconf import DictConfig
import seaborn as sns
import matplotlib.pyplot as plt

from src.baselines.SPD import SheafConnectionLaplacian
from src.baselines.SLGP import SmoothSheafDiffusion
from src.builder import *
from src.solver import *
from src.utils.dsp_utils import *

@hydra.main(
    config_path="../configs",  
    config_name="wind_inference",
    version_base="1.3"
)

def main(cfg: DictConfig):
    # Read data 
    X = np.load('data/important/PeakWeather/wind_data_uv_full.npy')

    V = X.shape[0] // 2
    d = 2

    train_split_idx = int(0.8 * X.shape[1])
    X_train = X[:, 0 : train_split_idx]
    X_test = X[:, train_split_idx:]

    C = ( 1 / X_train.shape[1] ) * X_train @ X_train.T

    # Sheaf Laplacian Learning
    solver = cfg.solver
    match solver:
        case 'KRON':
            # Kronecker Graph Estimation
            solver_ = SCGL(
                V = V,
                d = d,
                k = 1,
                alpha = cfg.solvers.SCGL.alpha,
                beta = cfg.solvers.SCGL.beta,
                initialization_mode = 'ID-QP',
                proximal_mode = cfg.solvers.SCGL.proximal_mode,
                update_frames = False,
                fix_beta = cfg.solvers.SCGL.fix_beta,
                beta_min = cfg.solvers.SCGL.beta_min,
                beta_max = cfg.solvers.SCGL.beta_max,
                beta_factor = cfg.solvers.SCGL.beta_factor,
                MAX_ITER=20000
            )

            sols = solver_.fit(X_train)
            w_, _ = sols["SCGL"]['w'], sols["SCGL"]['O']

            L_hat = LKron(w_, V, d) 

        case 'SCGL':
            # Connection Graph Estimation
            solver_ = SCGL(
                V = V,
                d = d,
                k = 1,
                alpha = cfg.solvers.SCGL.alpha,
                beta = cfg.solvers.SCGL.beta,
                initialization_mode = cfg.solvers.SCGL.initialization_mode,
                proximal_mode = cfg.solvers.SCGL.proximal_mode,
                update_frames = cfg.solvers.SCGL.update_frames,
                SOC = cfg.solvers.SCGL.SOC,
                rho = cfg.solvers.SCGL.rho,
                fix_beta = cfg.solvers.SCGL.fix_beta,
                beta_min = cfg.solvers.SCGL.beta_min,
                beta_max = cfg.solvers.SCGL.beta_max,
                beta_factor = cfg.solvers.SCGL.beta_factor,
                MAX_ITER=20000
            )

            sols = solver_.fit(X_train)

            w_, O_ = sols["SCGL"]['w'], sols["SCGL"]['O']

            L_hat = O_.T @ LKron(w_, V, d) @ O_

        case 'SCOP':
            # Connection Graph Estimation
            solver_ = SCGL(
                V = V,
                d = d,
                k = 1,
                alpha = cfg.solvers.SCGL.alpha,
                beta = cfg.solvers.SCGL.beta,
                initialization_mode = cfg.solvers.SCGL.initialization_mode,
                proximal_mode = cfg.solvers.SCGL.proximal_mode,
                update_frames = cfg.solvers.SCGL.update_frames,
                SOC = cfg.solvers.SCGL.SOC,
                rho = cfg.solvers.SCGL.rho,
                fix_beta = cfg.solvers.SCGL.fix_beta,
                beta_min = cfg.solvers.SCGL.beta_min,
                beta_max = cfg.solvers.SCGL.beta_max,
                beta_factor = cfg.solvers.SCGL.beta_factor,
            )

            w_, _, _, _, O_ , _ = solver_.SCGL_initialization(X_train)
            L_hat = O_.T @ LKron(w_, V, d) @ O_

        case 'SDP':
            # SPD Graph Estimation
            solver_ = SheafConnectionLaplacian(
                V = V,
                d = d,
                L0 = None
            )

            solver_.cross_validation(X_train, verbose = 0)
            L_hat = solver_.solve(C, verbose = 1)

        case 'SLGP':
            # SLGP Graph estimation
            L_hat = SmoothSheafDiffusion(X_train, V, d, int(1.1 * np.log(V) * (V - 1) / 2)).LaplacianBuilder()

    uuid : str = f'wind_{solver}.npy'
    SAVE_DIR : Path = Path('.') / 'data/interim/wind/' / uuid

    np.save(SAVE_DIR, L_hat)

if __name__ == '__main__':
    main()