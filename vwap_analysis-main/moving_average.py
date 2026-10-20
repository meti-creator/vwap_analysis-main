import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports]

df = pd.read_csv("nq_subset.csv")

print(df.columns) 
print("Loaded shape:", df.shape)
print("Data type of 'dt' column:", df["dt"].dtype)
converted_dt = pd.to_datetime(df["dt"], utc=True)
print("Data type of 'dt' column after conversion:", converted_dt.dtype)
df["dt"] = converted_dt
sorted_df = df.sort_values("dt").reset_index(drop=True)
print("Sorted shape:", sorted_df.shape)
print("Sorted data type of 'dt' column:", sorted_df["dt"].dtype)
gaps = sorted_df["dt"].diff().value_counts().head(10)
print("Time gaps between consecutive rows:\n", gaps)
df["dt"].diff()
# ...existing code...
sorted_df = df.sort_values("dt").reset_index(drop=True)
print("Sorted shape:", sorted_df.shape)
print("Sorted data type of 'dt' column:", sorted_df["dt"].dtype)

df = sorted_df
df["gap"] = df["dt"].diff()
df["block"] = (df["gap"] > pd.Timedelta("10min")).cumsum()

block_lengths = df.groupby("block").size()
print("Number of blocks:", len(block_lengths))
print("Shortest block length:", block_lengths.min())
print("Longest block length:", block_lengths.max())

gaps = sorted_df["dt"].diff().value_counts().head(10)
print("Time gaps between consecutive rows:\n", gaps)
