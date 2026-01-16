""" Script for experiment 3 in the journal paper """

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
    config_name="spheres",
    version_base="1.3"
)

def main(cfg: DictConfig):
    """ Main experimental loop"""
    # Read simulations specific variables
    V : int = cfg.dimensions.V
    d = 2
    ratio : float = cfg.signals.ratio
    train_seed : float = cfg.signals.train_seed

    # Data generation
    graph_ = FibonacciSphereGraph(V = V, k_neighbors= 4)
    X_train = graph_.random_tangent_bundle_signals(M = int(V * d * ratio), seed=train_seed)

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
            )

            sols = solver_.fit(X_train.X)
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
            )

            sols = solver_.fit(X_train.X)

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

            sols = solver_.fit(X_train.X)

            w_, O_ = sols["Initialization"]['w'], sols["Initialization"]['O']

            L_hat = O_.T @ LKron(w_, V, d) @ O_

        case 'SPD':
            # SPD Graph Estimation
            solver_ = SheafConnectionLaplacian(
                V = V,
                d = d,
                L0 = None
            )

            solver_.cross_validation(X_train.X, verbose = 0)
            L_hat = solver_.solve(X_train.covariance, verbose = 0)

        case 'SLGP':
            # SLGP Graph estimation
            L_hat = SmoothSheafDiffusion(X_train.X, V, d, len(graph_.edges)).LaplacianBuilder()

    uuid : str = f"{solver}.npy"
    SAVE_DIR : Path = Path('.') / 'data/interim/spheres/' / uuid

    np.save(SAVE_DIR, L_hat)
    
if __name__ == '__main__':
    main()



