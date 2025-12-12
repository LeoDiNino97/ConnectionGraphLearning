import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import pandas as pd
from pathlib import Path

# Define some usefull paths
CURRENT : Path = Path('.')
RESULTS_PATH : Path = CURRENT / 'data/interim'
SAVE_DIR : Path = CURRENT / 'data/processed'

SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Load all parquet files into a single DataFrame
dfs = []
for file in RESULTS_PATH.glob("*.parquet"):
    try:
        df = pd.read_parquet(file)
        dfs.append(df)
    except Exception as e:
        print(f"Skipping {file}: {e}")

# Combine
df_all = pd.concat(dfs, ignore_index=True)

# Define what to group by
group_cols = ["V", "d", "Ratio", "Solver", "Graph"]

# Define metric columns to average across seeds
metric_cols = ["F1", "Precision", "Recall", "Empirical Total Variation"]

# Aggregation
df_out = (
    df_all.groupby(group_cols)[metric_cols]
    .agg(['mean', 'std'])
)

# Flatten column index
df_out.columns = [
    f"{col}_{stat}" for col, stat in df_out.columns
]

df_out = df_out.reset_index()

df_out.to_parquet(SAVE_DIR / "random_graphs.parquet")





