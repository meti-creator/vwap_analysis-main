# TODO: build this stage
"""
STAGE 3: Compute the running (cumulative) VWAP and its volume-weighted
standard deviation band, resetting at the start of every session.

"""

import pandas as pd  # type: ignore[reportMissingModuleSource]


df = pd.read_csv("nq_tagged.csv")
df["dt"] = pd.to_datetime(df["dt"], utc=True)

print("Loaded shape:", df.shape)

df = df.sort_values("dt").reset_index(drop=True)

# 2. Compute typical price
# ---------------------------------------------------------------------------
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3


# 3. Compute cumulative VWAP, reset at every (trading_day, session) group
# ---------------------------------------------------------------------------


group_cols = ["trading_day", "session"]

# pv = price * volume for this candle -- the numerator ingredient
df["pv"] = df["typical_price"] * df["volume"]


df["cum_pv"] = df.groupby(group_cols)["pv"].cumsum()
df["cum_v"]  = df.groupby(group_cols)["volume"].cumsum()

# The running VWAP at every single candle 

df["vwap"] = df["cum_pv"] / df["cum_v"]

# ---------------------------------------------------------------------------
# 4. Compute the running volume-weighted standard deviation

df["vol_dev_sq"] = df["volume"] * (df["typical_price"] - df["vwap"]) ** 2
df["cum_vol_dev_sq"] = df.groupby(group_cols)["vol_dev_sq"].cumsum()

df["vwap_variance"] = df["cum_vol_dev_sq"] / df["cum_v"]
df["vwap_sd"] = df["vwap_variance"] ** 0.5


# 5. Sanity checks

sample_day = df["trading_day"].iloc[0]
sample = df[(df["trading_day"] == sample_day) & (df["session"] == "NewYork")]
print(f"\nSample NY session on {sample_day}:")
print(sample[["dt", "typical_price", "volume", "vwap", "vwap_sd"]].head(10))

print("\nAny NaN in vwap or vwap_sd?")
print(df[["vwap", "vwap_sd"]].isnull().sum())

# 6. Save

df.to_csv("nq_vwap.csv", index=False)
print("\nSaved nq_vwap.csv")