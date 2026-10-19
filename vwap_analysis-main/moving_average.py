import pandas as pd
import numpy as np

df = pd.read_csv("nq_vwap.csv")
df['MA'] = df['VWAP'].rolling(window=20).mean()