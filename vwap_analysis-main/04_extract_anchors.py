# TODO: build this stage
"""
STAGE 4: For every trading day, extract the FINAL New York session VWAP
value and SD -- this becomes the frozen anchor (V_n, sigma_n) that we'll
test the NEXT day's price action against in Stage 5.

Run from PowerShell with:
    python 04_extract_anchors.py
"""

import pandas as pd

# name the input file from Stage 3's output
# 1. Load Stage 3's output
# ------------------------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv("nq_vwap.csv")
df["dt"] = pd.to_datetime(df["dt"], utc=True)
df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date

print("Loaded shape:", df.shape)


# 2. Isolate just the New York session rows

ny = df[df["session"] == "NewYork"].copy()
print("NewYork session rows:", ny.shape)

# 3. For each trading day, grab the LAST row of that day's NY session

ny = ny.sort_values("dt")

anchors = ny.groupby("trading_day").tail(1)

# ---------------------------------------------------------------------------
# 4. Keep only the columns we actually need going forward
# ---------------------------------------------------------------------------
anchors = anchors[["trading_day", "vwap", "vwap_sd"]].rename(
    columns={"vwap": "V", "vwap_sd": "sigma"}
)

anchors["lower_band"] = anchors["V"] - anchors["sigma"]
anchors["upper_band"] = anchors["V"] + anchors["sigma"]

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. Sanity checks
# 
print("\nNumber of daily anchors:", len(anchors))
print("\nSample anchors:")
print(anchors.head(10))

print("\nAny NaN anchors? (would indicate a day with no NY session data)")
print(anchors.isnull().sum())

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
anchors.to_csv("daily_anchors.csv", index=False)
print("\nSaved daily_anchors.csv")