# TODO: build this stage
"""
STAGE 2: Tag every candle with:

"""

import pandas as pd


# 1. Load Stage 1's output

df = pd.read_csv("nq_subset.csv")


df["dt"] = pd.to_datetime(df["dt"], utc=True)

print("Loaded shape:", df.shape)


# 2. Define session boundaries (in UTC hours)
# session boundaries (in UTC hours) for the three main sessions:

#   Asia:      00:00 - 08:00 UTC
#   London:    08:00 - 13:00 UTC
#   New York:  13:00 - 21:00 UTC

def get_session(hour):
    """Given an hour (0-23, UTC), return which session it belongs to."""
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 21:
        return "NewYork"
    else:
        return "Other"   

df["session"] = df["dt"].dt.hour.apply(get_session)

# 3. Assign a "trading day" label

df["trading_day"] = df["dt"].dt.date

# 4. Drop rows outside our three defined sessions
before = len(df)
df = df[df["session"] != "Other"].copy()
after = len(df)
print(f"Dropped {before - after} rows outside defined sessions")

# 5. Sanity checks

print("\nRows per session:")
print(df["session"].value_counts())

print("\nNumber of unique trading days:", df["trading_day"].nunique())

print("\nSample rows:")
print(df[["dt", "session", "trading_day"]].head(10))


# 6. Save
df.to_csv("nq_tagged.csv", index=False)
print("\nSaved nq_tagged.csv")