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
    test_seed : float = cfg.signals.test_seed
    laplacian_operator : str = cfg.signals.laplacian_operator
    len_scale : float = cfg.signals.len_scale

    # Loading folder
    RESULTS_PATH: Path = Path('.') / 'data/interim/spheres/' 
    FIG_SAVE_DIR: Path = Path('.') / 'figs/spheres/'

    # Data generation
    graph_ = FibonacciSphereGraph(V = V, k_neighbors= 4)
    
    L_hat_KRON = np.load(RESULTS_PATH / f'KRON_len_scale{len_scale}.npy')
    L_hat_SCGL = np.load(RESULTS_PATH / f'SCGL_len_scale{len_scale}.npy')
    L_hat_SCOP = np.load(RESULTS_PATH / f'SCOP_len_scale{len_scale}.npy')
    L_hat_SLGP = np.load(RESULTS_PATH / f'SLGP_len_scale{len_scale}.npy')
    L_hat_SDP = np.load(RESULTS_PATH / f'SDP_len_scale{len_scale}.npy')

    # # Compression experiment 
    X_test = graph_.random_tangent_bundle_signals(M = 1000, len_scale=len_scale, seed=test_seed)

    _, U0 = np.linalg.eigh(graph_.laplacians[laplacian_operator])
    _, U1 = np.linalg.eigh(L_hat_KRON)
    _, U2 = np.linalg.eigh(L_hat_SCGL)
    _, U3 = np.linalg.eigh(L_hat_SCOP)
    _, U4 = np.linalg.eigh(L_hat_SLGP)
    _, U5 = np.linalg.eigh(L_hat_SDP)

    non_zero_coeffs = [10,20,30,40,50,60,]

    SCGL_ = {k: 0 for k in non_zero_coeffs}
    SLGP_ = {k: 0 for k in non_zero_coeffs}
    KRON_ = {k: 0 for k in non_zero_coeffs}
    SCOP_ = {k: 0 for k in non_zero_coeffs}
    SDP_ = {k: 0 for k in non_zero_coeffs}
    VDM_ = {k: 0 for k in non_zero_coeffs}

    for k in non_zero_coeffs:
        VDM_[k] = nonlinear_approximation(U0, X_test.X, k,)
        KRON_[k] = nonlinear_approximation(U1, X_test.X, k,)
        SCGL_[k] = nonlinear_approximation(U2, X_test.X, k,)
        SCOP_[k] = nonlinear_approximation(U3, X_test.X, k,)
        SLGP_[k] = nonlinear_approximation(U4, X_test.X, k,)
        SDP_[k] = nonlinear_approximation(U5, X_test.X, k,)

    # --- Global Style Settings ---
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'mathtext.fontset': 'stix',
        'font.size': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
    })

    # --- Color/Marker palette ---
    colors_def = [
        '#0072BD', '#D95319', '#7E2F8E', '#EDB120',
        '#77AC30', '#4DBEEE', '#A2142F'
    ]

    markers_list = ['o', '<', 's', '^', '>', 'P', 'D']  # all filled markers

    plot_dicts = {
        "VDM": VDM_,
        "SCOP": SCOP_,
        "KRON": KRON_,
        "SLGP": SLGP_,
        "SDP": SDP_,
        "SCGL": SCGL_,
    }

    records = []
    for method, values in plot_dicts.items():
        for x, mean in zip(non_zero_coeffs, values.values()):
            records.append({
                "Sparsity": x,
                "Mean": mean,
                "Method": method
            })
    res_df2 = pd.DataFrame.from_records(records)

    methods = list(plot_dicts.keys())
    markers = {m: markers_list[i % len(markers_list)] for i, m in enumerate(methods)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i, (method, df_sub) in enumerate(res_df2.groupby("Method")):
        df_sub = df_sub.sort_values("Sparsity")
        color = colors_def[i % len(colors_def)]
        marker = markers[method]

        ax.plot(
            df_sub["Sparsity"],
            df_sub["Mean"],
            label=method,
            color=color,
            marker=marker,
            markersize=12,
            linewidth=2.4,
        )

    ax.set_yscale('log')
    ax.set_ylabel(r"NMSE$(\mathbf{F},\hat{\mathbf{F}})$")
    ax.set_xlabel("Number of non-zero coefficients")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    plt.grid(True, which="minor", color="gray", lw=0.5, alpha=0.3)

    ax.legend(
        fontsize=14,
        frameon=True,
        loc="lower left",
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.0
    )

    plt.tight_layout()
    plt.savefig(FIG_SAVE_DIR / f"spheres_len_scale{len_scale}.pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()