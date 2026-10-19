"""
STAGE 5: The actual hypothesis test.

For every trading day n+1, test its NEW YORK session candles ONLY against
the PREVIOUS trading day's frozen NY anchor band [V_n - sigma_n, V_n + sigma_n].
(Asia and London session candles on day n+1 are excluded from this test --
this is a stricter, apples-to-apples NY-close-to-NY-session comparison.)

We compute three related but distinct metrics per day:
  1. Containment rate: % of candles whose typical_price fell inside the band
  2. Touch rate: % of candles whose high or low made any contact with the band
  3. Full breach flag: did the day's high/low EVER escape the band entirely


"""

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load Stage 3's candle data and Stage 4's daily anchors
# ---------------------------------------------------------------------------
candles = pd.read_csv("nq_vwap.csv")
candles["dt"] = pd.to_datetime(candles["dt"], utc=True)
candles["trading_day"] = pd.to_datetime(candles["trading_day"]).dt.date

anchors = pd.read_csv("daily_anchors.csv")
anchors["trading_day"] = pd.to_datetime(anchors["trading_day"]).dt.date
anchors = anchors.sort_values("trading_day").reset_index(drop=True)

print("Candles shape:", candles.shape)
print("Anchors shape:", anchors.shape)

# ---------------------------------------------------------------------------
# 2. Shift the anchor table forward by one TRADING day (not calendar day)

anchors["test_day"] = anchors["trading_day"].shift(-1)

# Drop the very last anchor -- it has no "next day" to test against yet.
anchors = anchors.dropna(subset=["test_day"])
anchors["test_day"] = pd.to_datetime(anchors["test_day"]).dt.date

anchor_lookup = anchors[["test_day", "V", "sigma", "lower_band", "upper_band"]]

# ---------------------------------------------------------------------------
# 3. Merge: attach the PREVIOUS day's anchor onto each candle's trading_day
# 
merged = candles.merge(
    anchor_lookup,
    left_on="trading_day",
    right_on="test_day",
    how="inner",   # drop candles with no previous-day anchor (e.g. very first day)
)

print("Merged shape (all sessions, before NY filter):", merged.shape)

# ---------------------------------------------------------------------------
# 3b. Restrict to NEW YORK session candles only

merged = merged[merged["session"] == "NewYork"].copy()

print("Merged shape (NY session only):", merged.shape)

# ---------------------------------------------------------------------------
# 3c.  save the raw CANDLE-LEVEL containment flags too

candle_level_cols = [
    "dt", "trading_day", "session", "typical_price",
    "lower_band", "upper_band",
]


# 4. Per-candle containment test (using typical_price, as we discussed)

merged["contained"] = (
    (merged["typical_price"] >= merged["lower_band"])
    & (merged["typical_price"] <= merged["upper_band"])
)

# 5. Per-candle breach test (using high/low, the stricter secondary metric)

# "Escaped" means the candle's range poked outside the band in either
# direction -- even briefly.
merged["escaped"] = (
    (merged["high"] > merged["upper_band"])
    | (merged["low"] < merged["lower_band"])
)


# 5b. Per-candle "touched" test (looser than contained -- any contact counts)

merged["touched_band"] = (
    ((merged["high"] >= merged["lower_band"]) & (merged["high"] <= merged["upper_band"]))
    | ((merged["low"] >= merged["lower_band"]) & (merged["low"] <= merged["upper_band"]))
    | merged["contained"]   # if typical_price itself is inside, that also counts as contact
)

# ---------------------------------------------------------------------------
# 6. Aggregate to ONE ROW PER DAY
# ---------------------------------------------------------------------------
daily_results = merged.groupby("trading_day").agg(
    total_candles=("contained", "size"),
    contained_candles=("contained", "sum"),
    touched_candles=("touched_band", "sum"),
    any_escape=("escaped", "any"),
    V=("V", "first"),
    sigma=("sigma", "first"),
).reset_index()

daily_results["pct_contained"] = (
    daily_results["contained_candles"] / daily_results["total_candles"]
)
daily_results["pct_touched"] = (
    daily_results["touched_candles"] / daily_results["total_candles"]
)

# "Fully contained day" = every single candle stayed inside the band --
# the strict version of your original ask ("number of days price stayed
# within 1 SD").
daily_results["fully_contained"] = daily_results["pct_contained"] == 1.0

# ---------------------------------------------------------------------------
# 7. Summary stats
# ---------------------------------------------------------------------------
print("\n--- SUMMARY ---")
print("Total days tested:", len(daily_results))
print("Days FULLY contained (100% of candles in band):", daily_results["fully_contained"].sum())
print("Days with at least one escape (high/low broke band):", daily_results["any_escape"].sum())
print("\nAverage % of candles contained per day (typical_price):", daily_results["pct_contained"].mean())
print("Median % of candles contained per day (typical_price):", daily_results["pct_contained"].median())
print("\nAverage % of candles that TOUCHED the band per day (high/low contact):", daily_results["pct_touched"].mean())
print("Median % of candles that TOUCHED the band per day:", daily_results["pct_touched"].median())

print("\nSample daily results:")
print(daily_results.head(10))

# ---------------------------------------------------------------------------
# 8. Save

daily_results.to_csv("containment_results.csv", index=False)
print("\nSaved containment_results.csv")

# Save the per-candle detail file (this WILL show the "contained" column
merged[candle_level_cols + ["contained", "escaped", "touched_band"]].to_csv(
    "candle_level_containment.csv", index=False
)
print("Saved candle_level_containment.csv (per-candle detail)")