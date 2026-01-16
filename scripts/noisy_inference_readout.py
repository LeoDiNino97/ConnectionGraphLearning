import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

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

    # Define some useful paths
    CURRENT: Path = Path('.')
    folder_uuid : str = f"graph{graph}_V{V}_d{d}_graphseed{graph_seed}_{cfg.solvers.SCGL.proximal_mode}_{cfg.solvers.SCGL.alpha}_{cfg.solvers.SCGL.beta}"
    RESULTS_PATH: Path = CURRENT / 'data/interim/noisy_inference/' / folder_uuid
    TABLE_SAVE_DIR: Path = CURRENT / 'data/processed/noisy_inference' / folder_uuid
    FIG_SAVE_DIR: Path = CURRENT / 'figs/noisy_inference' / folder_uuid

    TABLE_SAVE_DIR.mkdir(exist_ok=True, parents=True)
    FIG_SAVE_DIR.mkdir(exist_ok=True, parents=True)

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
    # --- Color / marker palette (journal-consistent) ---
    colors_def = [
        '#0072BD', '#D95319', '#7E2F8E', '#EDB120',
        '#77AC30', '#4DBEEE', '#A2142F'
    ]
    markers_list = ['o', '<', 's', '^', '>', 'P', 'D']

    # Load all parquet files
    dfs = []
    for file in RESULTS_PATH.glob("*.parquet"):
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    # Combine into a single DataFrame
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["SNR"] = pd.to_numeric(df_all["SNR"], errors="coerce")

    # Grouping columns (schemes grouped by SNR)
    group_cols = ["V", "d", "Ratio", "Solver", "Graph", "SNR"]

    # Metrics to average
    metric_cols = ["F1", "Precision", "Recall", "Empirical Total Variation", "Signal Error"]

    # Compute mean
    mean_out = (
        df_all
        .groupby(group_cols)[metric_cols]
        .mean()
        .reset_index()
    )

    sd_out = (
        df_all
        .groupby(group_cols)[metric_cols]
        .std()
        .reset_index()
    )
    
    # Merge mean and std (suffixes _mean and _std)
    df_stats = mean_out.merge(
        sd_out,
        on=group_cols,
        suffixes=("_mean", "_std")
    )

    # Round numeric columns
    numeric_cols = mean_out.select_dtypes(include="number").columns
    mean_out[numeric_cols] = mean_out[numeric_cols].round(6)

    # Save result
    df_stats.to_parquet(TABLE_SAVE_DIR / "noisy_inference.parquet")

    # Metrics to plot
    metrics_to_plot = ['F1', 'Empirical Total Variation', 'Signal Error']

    metrics_name = {
        'F1': 'F1 score',
        'Empirical Total Variation': 'Normalized Empirical Total Variation',
        'Signal Error': 'Signal Error'
    }

    metrics_label = {
        'F1': 'F1',
        'Empirical Total Variation': 'NETV',
        'Signal Error': 'NMSE IIR'
    }

    # Sort for consistent plotting
    df_plot = df_stats.sort_values(by=["Ratio", "SNR"])

    ratios = sorted(df_plot["Ratio"].unique())
    noise_SNRs = sorted(df_plot["SNR"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)

    snr_to_color = {
        snr: colors_def[i % len(colors_def)]
        for i, snr in enumerate(noise_SNRs)
    }
    snr_to_marker = {
        snr: markers_list[i % len(markers_list)]
        for i, snr in enumerate(noise_SNRs)
    }

    for i, (ax, metric) in enumerate(zip(axes, metrics_to_plot)):

        for noise in noise_SNRs:

            df_scgl = df_plot[
                (df_plot["SNR"] == noise) &
                (df_plot["Solver"] == "SCGL")
            ]

            mean_vals, std_vals = [], []

            for r in ratios:
                row = df_scgl[df_scgl["Ratio"] == r]
                mean_vals.append(row[f"{metric}_mean"].mean())
                std_vals.append(row[f"{metric}_std"].mean())

            mean_vals = np.array(mean_vals)
            std_vals = np.array(std_vals)

            color = snr_to_color[noise]
            marker = snr_to_marker[noise]

            ax.plot(
                ratios,
                mean_vals,
                label=f"{noise} dB",
                color=color,
                marker=marker,
                markersize=9,
                linewidth=2.4,
            )

            lower = (mean_vals - std_vals  ).clip(min=1e-12)
            upper = mean_vals + std_vals 

            ax.fill_between(
                ratios,
                lower,
                upper,
                color=color,
                alpha=0.2
            )

            # CONTROL baseline (Signal Error only)
            if metric == "Signal Error":
                df_ctrl = df_plot[
                    (df_plot["SNR"] == noise) &
                    (df_plot["Solver"] == "CONTROL")
                ]

                ctrl_means = [
                    df_ctrl[df_ctrl["Ratio"] == r][f"{metric}_mean"].mean()
                    for r in ratios
                ]

                ax.plot(
                    ratios,
                    ctrl_means,
                    linestyle="--",
                    linewidth=2.2,
                    color=color,
                    alpha=0.9
                )

        ax.set_xlabel("Sampling Ratio")
        ax.set_ylabel(metrics_label[metric])

        if i in [1, 2]:
            ax.set_yscale("log")
            ax.grid(True, which="minor", color="gray", lw=0.5, alpha=0.3)
        else:
            ax.grid(True, linestyle="--", alpha=0.6)

        # --- Spine styling (journal style) ---
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("black")

    # --- Shared legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="SNR",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(noise_SNRs),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(FIG_SAVE_DIR / "noisy_inference.pdf", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()