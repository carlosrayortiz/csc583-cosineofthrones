import pandas as pd
import os

# Path to your CSV (update if needed)
csv_path = "ragthrones/data/artifacts/got_aug_chunks.csv"

# Output PKL path
pkl_path = "ragthrones/data/artifacts/df_aug.pkl"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

print(f"Loading CSV from: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Saving PKL to: {pkl_path}")
df.to_pickle(pkl_path)

print("Conversion complete.")
print(f"Rows: {len(df)}")