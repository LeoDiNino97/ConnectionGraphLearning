"""
PNN experiment preprocessing script (Journal Paper)
Processes HCP concatenated task data into zone × task averaged features.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(sys.path[0]).parent))

import pandas as pd
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

# ============================================================
# Configuration
# ============================================================

label_list = ['EMO', 'GAM', 'LAN', 'MOT', 'REL', 'SOC', 'WM']

onset_tasks = np.array([300, 444, 600, 873, 1043, 1181, 1309, 1621])
label_zones = np.array([1,10,16,24,31,34,38,51,59,67,74,79,81,90,100]) - 1

root_path = Path('data/important/HCP/HCP_concatenated_task')
save_path = Path('data/important/HCP/ready_to_go')

(save_path / 'LR').mkdir(parents=True, exist_ok=True)
(save_path / 'RL').mkdir(parents=True, exist_ok=True)

meta_path = Path('data/important/HCP/HCP_YA_subjects_2026_02_20_16_30_04.csv')
meta = pd.read_csv(meta_path)

print(4 == np.array([1,1,1]))
# ============================================================
# Derived dimensions 
# ============================================================

n_zone_intervals = len(label_zones) - 1
n_task_intervals = len(onset_tasks) - 1
feature_dim = n_zone_intervals * n_task_intervals

age_bins = np.array(meta['Age'].unique())
age = np.zeros((3,100))
# Count subjects dynamically
subject_folders = [
    f for f in root_path.iterdir()
    if f.is_dir() 
    and f.name != 'Misc' 
    and int(f.name) in np.array(meta['Subject']) 
    and meta[meta['Subject'] == int(f.name)]['Age'].values[0] != age_bins[-1]
]
age_bins = age_bins[:-1]
n_subjects = len(subject_folders)

LR_processed = np.zeros((feature_dim, n_subjects))
RL_processed = np.zeros((feature_dim, n_subjects))


# ============================================================
# Processing loop
# ============================================================

for k, folder in enumerate(sorted(subject_folders)):
    print(f"Processing subject {k+1}/{n_subjects}: {folder.name}")

    # Load data
    LR = loadmat(
        folder / 'TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_LR.mat'
    )['TS']

    RL = loadmat(
        folder / 'TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_RL.mat'
    )['TS']

    # Sanity check
    assert LR.shape == RL.shape, "LR and RL shapes mismatch."

    # Allocate subject feature matrices (interval-based!)
    LR_sub = np.zeros((n_zone_intervals, n_task_intervals))
    RL_sub = np.zeros((n_zone_intervals, n_task_intervals))

    # Compute interval averages
    for i in range(n_task_intervals):
        t_start, t_end = onset_tasks[i], onset_tasks[i+1]

        for j in range(n_zone_intervals):
            z_start, z_end = label_zones[j], label_zones[j+1]

            LR_sub[j, i] = np.mean(LR[z_start:z_end, t_start:t_end])
            RL_sub[j, i] = np.mean(RL[z_start:z_end, t_start:t_end])
    
    # Save per-subject matrices
    np.save(save_path / 'LR' / f'{folder.name}.npy', LR_sub)
    np.save(save_path / 'RL' / f'{folder.name}.npy', RL_sub)

    # Flatten for full dataset matrix
    LR_processed[:, k] = LR_sub.reshape(-1)
    RL_processed[:, k] = RL_sub.reshape(-1)
    print(meta[meta['Subject'] == int(folder.name)]['Age'])
    age[:,k] = (age_bins == meta[meta['Subject'] == int(folder.name)]['Age'].values[0])

# ============================================================
# Save full dataset matrices
# ============================================================
save_path_final = Path('data/processed/HCP')

np.save(save_path_final / 'LR_full.npy', LR_processed)
np.save(save_path_final / 'RL_full.npy', RL_processed)
np.save(save_path_final / 'age.npy', age)
print("\nProcessing complete.")
print(f"Feature dimension per subject: {feature_dim}")
print(f"Number of subjects: {n_subjects}")


