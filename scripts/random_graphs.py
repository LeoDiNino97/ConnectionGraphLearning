import sys
from pathlib import Path

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
    config_path = '../configs',
    config_name = 'random_graphs',
    version_base = '1.3',
)
def main(cfg: DictConfig) -> None:
    """Main experimental loop"""

    # Define some usefull paths
    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'data/interim'

    # Create directories
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # Read simulations specific variables
    V : int = cfg.dimensions.V
    d : int = cfg.dimensions.d
    ratio : float = cfg.dimensions.float
    noisy : bool = cfg.dimensions.noisy
    seed : int = cfg.dimensions.seed

    # Graph istantiation 
    graph = cfg.graph
    match graph:
        case 'ER':
            graph_ = ERCG(
                V = V,
                d = d,
                k = cfg.graphs.ER.k,
                seed = seed
            )

        case 'RBF':
            graph_ = RBFCG(
                V = V,
                d = d,
                sigma = cfg.graphs.RBF.sigma,
                cutoff = cfg.graphs.RBF.cutoff,
                seed = seed
            )
        
        case 'SBM':
            graph_ = SBMCG(
                V = V,
                d = d,
                seed = seed,
                k = len(cfg.graphs.SBD.p_k),
                p_k = cfg.graphs.SBM.p_k,
                p_in = cfg.graphs.SBM.p_in,
                p_out = cfg.graphs.SBM.p_out,
            )
        
    X = graph_.cochains_sampling(
        N = int(V * d * ratio),
        seed = seed,
        noisy = noisy
    )

    # Solvers istantiation and laplacian learning
    solver = cfg.solver
    match solver:
        case 'SCGL':
            solver_ = SCGL(
                V = V,
                d = d,
                k = cfg.solvers.SCGL.k,
                alpha = cfg.solvers.SCGL.alpha,
                beta = cfg.solvers.SCGL.beta,
                initialization_mode = cfg.solvers.SCGL.initialization_mode,
                update_frames = cfg.solvers.SCGL.update_frames,
                SOC = cfg.solvers.SCGL.SOC,
                rho = cfg.solvers.SCGL.rho,
                fix_beta = cfg.solvers.SCGL.fix_beta,
                beta_min = cfg.solvers.SCGL.beta_min,
                beta_max = cfg.solvers.SCGL.beta_max,
                beta_factor = cfg.solvers.SCGL.beta_factor,
            )
            O, w, _, _ = solver_.fit(X.X)
            L_hat = O.T @ LKron(w, V, d) @ O

        case 'SPD':
            solver_ = SheafConnectionLaplacian(
                V = V,
                d = d,
            )

            # Initializing the SPD solver and validating parameter alpha via cross validation
            solver_.L0 = graph_.CL
            solver_.CrossValidation(X.X, verbose = 0)
            L_hat = solver_.Solve(X.covariance, verbose = 0)

        case 'SLGP':
            L_hat = SmoothSheafDiffusion(X.X, V, d, len(graph.edges)).LaplacianBuilder()

    uuid : str = f"V{V}_d{d}_seed{seed}_ratio{ratio}_{solver}_{graph}"

    # Collecting metrics
    # Test signals for total variation
    X_test = graph_.CochainsSampling(int(d * V * ratio),)

    # Ground truth
    w_true = L_inv(graph.L)
    w_true_bin = (w_true > 0).astype(int)
    
    # Extracting sparsity pattern from Laplacian estimation 
    w_hat_bin = L_spy(L_hat, d)
    
    f1_score_ = f1_score(w_true_bin, w_hat_bin)
    precision_ = precision_score(w_true_bin, w_hat_bin)
    recall_ = recall_score(w_true_bin, w_hat_bin)
    empirical_TV = np.trace(L_hat @ X_test.covariance) 

    pd.DataFrame({
        'V': V,
        'd': d, 
        'Ratio': ratio,
        'Seed': seed,
        'Solver': solver,
        'Graph': graph,
        'F1': f1_score_,
        'Precision': precision_,
        'Recall': recall_,
        'Empirical Total Variation': empirical_TV
    }).to_parquet(RESULTS_PATH / f'{uuid}.parquet')

if __name__ == '__main__':
    main()

