"""
STAGE 6: The significance test + final report.

We already know the OBSERVED containment rate (from containment_results.csv,
NY session only). Now we build a NULL baseline using a permutation test:
shuffle which anchor band gets tested against which day (breaking the true
"yesterday" pairing), and see what containment rate that produces by pure
chance. Repeat thousands of times to build a distribution, then see where
our real result falls.

Scope note: this matches Stage 5's updated scope -- we're testing NEXT-DAY
NEW YORK SESSION candles ONLY against the previous day's frozen NY anchor,
not the full Asia+London+NY day.

Run from PowerShell with:
    python 06_report.py

NOTE: this script re-derives everything from nq_vwap.csv + daily_anchors.csv
rather than just re-reading containment_results.csv, so it can efficiently
re-run the containment test thousands of times during the permutation loop.
"""

import pandas as pd
import numpy as np

RNG = np.random.default_rng(42)   # fixed seed so results are reproducible
N_PERMUTATIONS = 500   # reduced from 2000 since local-window shuffling is more expensive per iteration

# ---------------------------------------------------------------------------
# 1. Load candle data, keep only what we need, and pre-sort per day
# ---------------------------------------------------------------------------
# IMPORTANT: Stage 5 was updated to test NY session candles ONLY (not all
# sessions), so this permutation test must match that same scope -- otherwise
# we'd be comparing our real result against a differently-scoped baseline.
candles = pd.read_csv("nq_vwap.csv", usecols=["dt", "trading_day", "session", "typical_price"])
candles["trading_day"] = pd.to_datetime(candles["trading_day"]).dt.date
candles = candles[candles["session"] == "NewYork"].copy()

# Build a dictionary: trading_day -> sorted numpy array of that day's prices.
# Sorting once up front lets us use np.searchsorted() later, which is a fast
# way to count "how many values fall between two bounds" without looping
# through every candle by hand each time.
day_prices = {
    day: np.sort(group["typical_price"].to_numpy())
    for day, group in candles.groupby("trading_day")
}

print("Days with candle data:", len(day_prices))

# ---------------------------------------------------------------------------
# 2. Load anchors, build the TRUE (real, chronological) test-day mapping
# ---------------------------------------------------------------------------
anchors = pd.read_csv("daily_anchors.csv")
anchors["trading_day"] = pd.to_datetime(anchors["trading_day"]).dt.date
anchors = anchors.sort_values("trading_day").reset_index(drop=True)

anchors["test_day"] = anchors["trading_day"].shift(-1)
anchors = anchors.dropna(subset=["test_day"]).reset_index(drop=True)
anchors["test_day"] = pd.to_datetime(anchors["test_day"]).dt.date

# Keep only anchors whose test_day actually has candle data available
anchors = anchors[anchors["test_day"].isin(day_prices.keys())].reset_index(drop=True)

test_days   = anchors["test_day"].to_numpy()
lower_bands = anchors["lower_band"].to_numpy()
upper_bands = anchors["upper_band"].to_numpy()

n_days = len(anchors)
print("Days used in permutation test:", n_days)

# ---------------------------------------------------------------------------
# 3. A fast function: given arrays of test_days + bands, compute the
#    average pct_contained across all days
# ---------------------------------------------------------------------------
def mean_containment(test_days_arr, lower_arr, upper_arr):
    """For each (day, band) pair, count what fraction of that day's candles
    fall inside [lower, upper], using fast sorted-array lookups. Returns the
    AVERAGE fraction across all days -- this is our one summary statistic."""
    pct_list = []
    for day, lo, hi in zip(test_days_arr, lower_arr, upper_arr):
        prices = day_prices[day]
        # searchsorted finds the insertion points for lo and hi in the
        # sorted array -- the difference between them is exactly the COUNT
        # of prices that fall between lo and hi. Much faster than looping.
        count_inside = np.searchsorted(prices, hi, side="right") - np.searchsorted(prices, lo, side="left")
        pct_list.append(count_inside / len(prices))
    return np.mean(pct_list)

# ---------------------------------------------------------------------------
# 4. Observed (TRUE, real chronological order) result
# ---------------------------------------------------------------------------
observed_mean = mean_containment(test_days, lower_bands, upper_bands)
print("\nObserved mean containment (true previous-day pairing):", observed_mean)

# ---------------------------------------------------------------------------
# 5. Build the NULL distribution via LOCAL BLOCK permutation
# ---------------------------------------------------------------------------
# IMPORTANT FIX: shuffling across the ENTIRE 6+ year dataset is flawed --
# NQ price trended enormously over that span (roughly 8,800 in 2020 to a
# much higher level by 2026), so a random anchor from a totally different
# year almost never overlaps with a given test day's price range. That
# makes the null distribution collapse toward ~0% containment almost by
# construction -- not because there's nothing special about "yesterday,"
# but simply because ANY two nearby days share a similar price level while
# distant ones don't. That's a trivial, uninteresting effect we don't want
# to accidentally take credit for.
#
# The fix: only shuffle anchors WITHIN a local window (here, +/- WINDOW
# trading days) around each test day's true position in the sequence. This
# way, both the real pairing and every random pairing draw from roughly the
# same price era -- so the test now isolates whether YESTERDAY specifically
# beats other NEARBY days, which is the actual question the assignment asks.
WINDOW = 10  # trading days on each side to draw random "fake" anchors from

null_means = np.empty(N_PERMUTATIONS)
n = n_days

for i in range(N_PERMUTATIONS):
    shuffled_lower = np.empty(n)
    shuffled_upper = np.empty(n)
    for idx in range(n):
        lo_idx = max(0, idx - WINDOW)
        hi_idx = min(n, idx + WINDOW + 1)
        # Pick a random day WITHIN the local window, excluding the true
        # previous day itself (idx), so we're comparing against genuine
        # alternatives, not accidentally re-including the real answer.
        candidates = [j for j in range(lo_idx, hi_idx) if j != idx]
        pick = candidates[RNG.integers(0, len(candidates))]
        shuffled_lower[idx] = lower_bands[pick]
        shuffled_upper[idx] = upper_bands[pick]
    null_means[i] = mean_containment(test_days, shuffled_lower, shuffled_upper)
    if (i + 1) % 50 == 0:
        print(f"  permutation {i + 1}/{N_PERMUTATIONS} done")

print("\nNull distribution (random pairing) summary:")
print("  mean:", null_means.mean())
print("  std: ", null_means.std())
print("  min: ", null_means.min())
print("  max: ", null_means.max())

# ---------------------------------------------------------------------------
# 6. p-value: how extreme is our observed result vs the null distribution?
# ---------------------------------------------------------------------------
# One-sided test: we're specifically asking "is the TRUE pairing's
# containment rate HIGHER than random pairing produces?" (that's the
# gravitation hypothesis). So we count how many null permutations did
# AT LEAST AS WELL as our real result -- a small fraction means our real
# result is unusually high, i.e. statistically significant.
p_value = (np.sum(null_means >= observed_mean) + 1) / (N_PERMUTATIONS + 1)

z_score = (observed_mean - null_means.mean()) / null_means.std()

print("\n--- SIGNIFICANCE TEST RESULT ---")
print(f"Observed mean containment: {observed_mean:.4f}")
print(f"Null mean containment:     {null_means.mean():.4f}")
print(f"Null std:                  {null_means.std():.4f}")
print(f"Z-score:                   {z_score:.2f}")
print(f"P-value (one-sided):       {p_value:.4f}")

if p_value < 0.05:
    verdict = "STATISTICALLY SIGNIFICANT (p < 0.05) -- real pairing beats random chance"
else:
    verdict = "NOT statistically significant (p >= 0.05) -- indistinguishable from random chance"
print(f"Verdict: {verdict}")

# ---------------------------------------------------------------------------
# 7. Save results for the report
# ---------------------------------------------------------------------------
np.savetxt("null_distribution.csv", null_means, delimiter=",", header="null_mean_containment", comments="")

with open("final_report.md", "w") as f:
    f.write("# VWAP Gravitation Analysis -- Final Report\n\n")
    f.write("Scope: next-day NEW YORK session candles only, tested against the previous day's frozen NY anchor band.\n\n")
    f.write(f"Null model: LOCAL block permutation (window = +/-{WINDOW} trading days) -- ")
    f.write("random 'fake' anchors are drawn only from nearby days, so the test isolates whether\n")
    f.write("YESTERDAY specifically outperforms other nearby days, controlling for the trivial fact\n")
    f.write("that nearby days naturally share similar price levels regardless of any VWAP effect.\n\n")
    f.write(f"- Trading days tested: {n_days}\n")
    f.write(f"- Observed mean containment (true previous-day pairing): {observed_mean:.4f}\n")
    f.write(f"- Null mean containment (random pairing, {N_PERMUTATIONS} permutations): {null_means.mean():.4f}\n")
    f.write(f"- Null std: {null_means.std():.4f}\n")
    f.write(f"- Z-score: {z_score:.2f}\n")
    f.write(f"- P-value (one-sided): {p_value:.4f}\n")
    f.write(f"- Verdict: {verdict}\n")

print("\nSaved null_distribution.csv and final_report.md")