"""
STEP 1: Load the raw data, verify it's correct, and trim it down to a
manageable size before we do any real analysis.

"""


import pandas  as pd

df = pd.read_csv("data-2020.csv")

print("Raw file shape:", df.shape)   # (num_rows, num_columns) -- quick sanity check
print(df.head())                     # show first 5 rows, confirm columns look right

# timezone-aware datetime in UTC, from the epoch seconds in ts_event
# 2. Convert ts_event (epoch seconds) into a real, timezone-aware datetime

df["dt"] = pd.to_datetime(df["ts_event"], unit="s", utc=True)


# 3. Sanity check: does the date range make sense?

print("Earliest timestamp:", df["dt"].min())
print("Latest timestamp:  ", df["dt"].max())

# 4. Check the bar interval (is this really 1-minute data?)

time_diffs = df["dt"].diff().dropna()
print("Median time between bars:", time_diffs.median())

subset = df.sort_values("dt").drop_duplicates(subset=["dt"]).copy()

print("Full cleaned shape:", subset.shape)
print("Full cleaned range:", subset["dt"].min(), "to", subset["dt"].max())


print("Null counts:\n", subset[["open", "high", "low", "close", "volume"]].isnull().sum())


# 6. Save the cleaned file

subset.to_csv("nq_subset.csv", index=False)
print("Saved nq_subset.csv")
