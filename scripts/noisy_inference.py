""" Script for experiment 2 in the journal paper """

import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score

from src.baselines.SPD import SheafConnectionLaplacian
from src.baselines.SLGP import SmoothSheafDiffusion
from src.builder import *
from src.solver import *

@hydra.main(
    config_path="../configs",  
    config_name="noisy_inference",
    version_base="1.3"
)
def main(cfg: DictConfig):
    """Main experimental loop"""

    # Read simulations specific variables
    V : int = cfg.dimensions.V
    d : int = cfg.dimensions.d
    graph_seed : int = cfg.dimensions.seed
    graph : str = cfg.graph

    ratio : float = cfg.signals.ratio
    noisy : bool = cfg.signals.noisy
    signals_seed : int = cfg.signals.seed
    SNR : float = cfg.signals.SNR

    # Define some useful paths
    CURRENT: Path = Path('.')
    folder_uuid : str = f"graph{graph}_V{V}_d{d}_graphseed{graph_seed}_{cfg.solvers.SCGL.proximal_mode}_{cfg.solvers.SCGL.alpha}_{cfg.solvers.SCGL.beta}"
    RESULTS_PATH: Path = CURRENT / 'data/interim/noisy_inference/' / folder_uuid

    # Create directories
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    
    # Check for consistency
    if SNR == 'None':
        noisy = False

    # Graph istantiation 
    match graph:
        case 'ER':
            graph_ = ERCG(
                V = V,
                d = d,
                k = cfg.graphs.ER.k,
                seed = graph_seed
            )

        case 'RBF':
            graph_ = RBFCG(
                V = V,
                d = d,
                sigma = cfg.graphs.RBF.sigma,
                cutoff = cfg.graphs.RBF.cutoff,
                seed = graph_seed
            )
        
        case 'SBM':
            graph_ = SBMCG(
                V = V,
                d = d,
                seed = graph_seed,
                k = len(cfg.graphs.SBM.p_k),
                p_k = cfg.graphs.SBM.p_k,
                p_in = cfg.graphs.SBM.p_in,
                p_out = cfg.graphs.SBM.p_out,
            )

    X = graph_.cochains_sampling(
        N = int(V * d * ratio),
        seed = signals_seed,
        noisy = noisy,
        SNR = SNR
    )

    # Solvers istantiation and laplacian learning
    solver = cfg.solver
    match solver:

        case 'KRON':
            solver_ = SCGL(
                V = V,
                d = d,
                k = np.sum(np.isclose(np.linalg.eigvalsh(graph_.L),0)),
                alpha = cfg.solvers.SCGL.alpha,
                beta = cfg.solvers.SCGL.beta,
                initialization_mode = 'ID-QP',
                proximal_mode = cfg.solvers.SCGL.proximal_mode,
                update_frames = False,
                noisy = cfg.signals.noisy,
                fix_beta = cfg.solvers.SCGL.fix_beta,
                beta_min = cfg.solvers.SCGL.beta_min,
                beta_max = cfg.solvers.SCGL.beta_max,
                beta_factor = cfg.solvers.SCGL.beta_factor,
            )

            sols = solver_.fit(X.X)
            w, O = sols["SCGL"]['w'], sols["SCGL"]['O']

            L_hat = O.T @ LKron(w, V, d) @ O
            w_hat_bin = (w > 0).astype(int)

        case 'SCGL':
            solver_ = SCGL(
                V = V,
                d = d,
                k = np.sum(np.isclose(np.linalg.eigvalsh(graph_.L),0)),
                alpha = cfg.solvers.SCGL.alpha,
                beta = cfg.solvers.SCGL.beta,
                initialization_seed = cfg.dimensions.seed,
                initialization_mode = cfg.solvers.SCGL.initialization_mode,
                proximal_mode = cfg.solvers.SCGL.proximal_mode,
                update_frames = cfg.solvers.SCGL.update_frames,
                max_w_its = cfg.solvers.SCGL.max_w_its,
                SOC = cfg.solvers.SCGL.SOC,
                rho = cfg.solvers.SCGL.rho,
                noisy = cfg.signals.noisy,
                fix_beta = cfg.solvers.SCGL.fix_beta,
                beta_min = cfg.solvers.SCGL.beta_min,
                beta_max = cfg.solvers.SCGL.beta_max,
                beta_factor = cfg.solvers.SCGL.beta_factor,
            )

            sols = solver_.fit(X.X)

            w, O = sols["SCGL"]['w'], sols["SCGL"]['O']
            w[w <= 0.05] = 0

            L_hat = O.T @ LKron(w, V, d) @ O
        
            w_hat_bin = (w > 0).astype(int)

        case 'SPD':
            solver_ = SheafConnectionLaplacian(
                V = V,
                d = d,
            )

            # Initializing the SPD solver and validating parameter alpha via cross validation
            solver_.cross_validation(X.X, verbose = 0)

            L_hat = solver_.solve(X.covariance, verbose = 0)
            w_hat_bin = L_spy(L_hat, d)

        case 'SLGP':
            L_hat = SmoothSheafDiffusion(X.X, V, d, len(graph_.edges)).LaplacianBuilder()
            w_hat_bin = L_spy(L_hat, d)

        case 'CONTROL':
            L_hat = graph_.CL
            w_hat_bin = L_spy(L_hat, d)

    uuid : str = f"V{V}_d{d}_graphseed{graph_seed}_signalseed{signals_seed}_SNR{SNR}_ratio{ratio}_{solver}_{graph}_{cfg.solvers.SCGL.proximal_mode}_{cfg.solvers.SCGL.alpha}_{cfg.solvers.SCGL.beta}_{datetime.today().strftime("%Y%m%d")}"

    # Collecting metrics
    # Test signals for total variation
    k_0 = np.sum(np.isclose(np.linalg.eigvalsh(graph_.L),0))
    X_test = graph_.cochains_sampling(1000, noisy=False)

    if noisy:
        X_test_noisy = graph_.cochains_sampling(1000, noisy=True, SNR=SNR)
        gamma_test = 1 / (2 * np.mean(np.linalg.eigvalsh(X_test_noisy.covariance)[0 : d * k_0]))

    # Ground truth
    w_true = L_inv(graph_.L)
    w_true_bin = (w_true > 0).astype(int)

    f1_score_ = f1_score(w_true_bin, w_hat_bin)
    precision_ = precision_score(w_true_bin, w_hat_bin)
    recall_ = recall_score(w_true_bin, w_hat_bin)
    empirical_TV = np.abs(np.trace(L_hat @ X_test.covariance) - d * (V - k_0)) / (d * (V - k_0))
    if noisy:
        signal_NMSE = np.linalg.norm(np.linalg.inv(L_hat + gamma_test * np.eye(d * V)) @ (gamma_test * X_test_noisy.X) - X_test.X, ord='fro') ** 2 / np.linalg.norm(X_test.X, ord='fro') ** 2
    else:
        signal_NMSE = 0
        
    pd.DataFrame({
        'V': [V],
        'd': [d], 
        'Ratio': [ratio],
        'Graph Seed': [graph_seed],
        'Signal Seed': [signals_seed],
        'SNR': [SNR],
        'Solver': [solver],
        'Graph': [graph],
        'F1': [f1_score_],
        'Precision': [precision_],
        'Recall': [recall_],
        'Empirical Total Variation': [empirical_TV],
        'Signal Error': [signal_NMSE]
    }).to_parquet(RESULTS_PATH / f'{uuid}.parquet')

if __name__ == '__main__': 
    main()

