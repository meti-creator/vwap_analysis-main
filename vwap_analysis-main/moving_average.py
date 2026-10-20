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

df["fast"] = df.groupby("block")["close"].transform(
    lambda values: values.rolling(window=20).mean()
)
df["slow"] = df.groupby("block")["close"].transform(
    lambda values: values.rolling(window=50).mean()
)

first_block = df["block"].iloc[0]
print(
    df.loc[df["block"] == first_block, ["dt", "block", "close", "fast", "slow"]]
    .head(12)
)

df["fast_above"] = df["fast"] > df["slow"]
df["fast_below"] = df["fast"] < df["slow"]

previous_fast_below = df.groupby("block")["fast_below"].shift(1, fill_value=False)
previous_fast_above = df.groupby("block")["fast_above"].shift(1, fill_value=False)

df["up_cross"] = df["fast_above"] & previous_fast_below
df["down_cross"] = df["fast_below"] & previous_fast_above

print("Up crossovers:", df["up_cross"].sum())
print("Down crossovers:", df["down_cross"].sum())
idx = df.index[df["up_cross"]][0]   # position of the first up crossover
print(df.loc[idx-3:idx+3, ["dt", "close", "fast", "slow", "fast_above", "up_cross"]])
...

TARGET = 0.005

def test_pair(data, fast_n, slow_n, n_ahead):
    data = data.copy()

    data["fast"] = data.groupby("block")["close"].transform(
        lambda values: values.rolling(window=fast_n).mean()
    )
    data["slow"] = data.groupby("block")["close"].transform(
        lambda values: values.rolling(window=slow_n).mean()
    )

    data["fast_above"] = data["fast"] > data["slow"]
    data["fast_below"] = data["fast"] < data["slow"]
    previous_below = data.groupby("block")["fast_below"].shift(1, fill_value=False)
    previous_above = data.groupby("block")["fast_above"].shift(1, fill_value=False)
    data["up_cross"] = data["fast_above"] & previous_below
    data["down_cross"] = data["fast_below"] & previous_above

    data["next_high"] = data.groupby("block")["high"].transform(
        lambda values: values.shift(-1).iloc[::-1]
        .rolling(window=n_ahead, min_periods=n_ahead).max().iloc[::-1]
    )
    data["next_low"] = data.groupby("block")["low"].transform(
        lambda values: values.shift(-1).iloc[::-1]
        .rolling(window=n_ahead, min_periods=n_ahead).min().iloc[::-1]
    )

    data["hit_up"] = (data["next_high"] / data["close"] - 1) >= TARGET
    data["hit_down"] = (1 - data["next_low"] / data["close"]) >= TARGET

    has_future = data["next_high"].notna() & data["next_low"].notna()
    up_events = data["up_cross"] & has_future
    down_events = data["down_cross"] & has_future
    valid = has_future & data["slow"].notna()

    return (
        int(up_events.sum()),
        round(data.loc[up_events, "hit_up"].mean() * 100, 2),
        round(data.loc[valid, "hit_up"].mean() * 100, 2),
        int(down_events.sum()),
        round(data.loc[down_events, "hit_down"].mean() * 100, 2),
        round(data.loc[valid, "hit_down"].mean() * 100, 2),
    )

def direction_check(fast_n, slow_n, n_ahead):
    data = df.copy()

    data["fast"] = data.groupby("block")["close"].transform(
        lambda values: values.rolling(window=fast_n).mean()
    )
    data["slow"] = data.groupby("block")["close"].transform(
        lambda values: values.rolling(window=slow_n).mean()
    )

    data["fast_above"] = data["fast"] > data["slow"]
    data["fast_below"] = data["fast"] < data["slow"]
    previous_below = data.groupby("block")["fast_below"].shift(1, fill_value=False)
    previous_above = data.groupby("block")["fast_above"].shift(1, fill_value=False)
    data["up_cross"] = data["fast_above"] & previous_below
    data["down_cross"] = data["fast_below"] & previous_above

    data["next_high"] = data.groupby("block")["high"].transform(
        lambda values: values.shift(-1).iloc[::-1]
        .rolling(window=n_ahead, min_periods=n_ahead).max().iloc[::-1]
    )
    data["next_low"] = data.groupby("block")["low"].transform(
        lambda values: values.shift(-1).iloc[::-1]
        .rolling(window=n_ahead, min_periods=n_ahead).min().iloc[::-1]
    )

    data["hit_up"] = (data["next_high"] / data["close"] - 1) >= TARGET
    data["hit_down"] = (1 - data["next_low"] / data["close"]) >= TARGET

    has_future = data["next_high"].notna() & data["next_low"].notna()
    up_events = data["up_cross"] & has_future
    down_events = data["down_cross"] & has_future
    valid = has_future & data["slow"].notna()

    return {
        "fast": fast_n,
        "slow": slow_n,
        "window": n_ahead,
        "up_cross_then_rise": round(data.loc[up_events, "hit_up"].mean() * 100, 2),
        "up_cross_then_fall": round(data.loc[up_events, "hit_down"].mean() * 100, 2),
        "down_cross_then_fall": round(data.loc[down_events, "hit_down"].mean() * 100, 2),
        "down_cross_then_rise": round(data.loc[down_events, "hit_up"].mean() * 100, 2),
        "base_rise": round(data.loc[valid, "hit_up"].mean() * 100, 2),
        "base_fall": round(data.loc[valid, "hit_down"].mean() * 100, 2),
    }

split_date = df["dt"].quantile(0.8)
early = df[df["dt"] <= split_date].copy()
late = df[df["dt"] > split_date].copy()
print("Split date:", split_date)

pairs = [(20, 200), (30, 200), (10, 200)]
rows = []

for period, part in [("early", early), ("late", late)]:
    for fast_n, slow_n in pairs:
        up_n, up_hit, up_base, down_n, down_hit, down_base = test_pair(
            part, fast_n, slow_n, 60
        )
        rows.append({
            "period": period,
            "fast": fast_n,
            "slow": slow_n,
            "window": 60,
            "up_count": up_n,
            "up_hit": up_hit,
            "up_base": up_base,
            "up_edge": round(up_hit - up_base, 2),
            "down_count": down_n,
            "down_hit": down_hit,
            "down_base": down_base,
            "down_edge": round(down_hit - down_base, 2),
        })

results = pd.DataFrame(rows)
print(results.to_string(index=False))

comparison = results.pivot(
    index=["fast", "slow", "window"],
    columns="period",
    values=["up_edge", "down_edge"],
)
print("\nEarly vs. late edge comparison:")
print(comparison.to_string())
direction_rows = [direction_check(f, 200, 60) for f in [10, 20, 30]]
direction = pd.DataFrame(direction_rows)
print("\nDirection check (window 60):")
print(direction.to_string(index=False))
