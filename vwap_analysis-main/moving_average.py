import pandas as pd
import numpy as np

df = pd.read_csv("data-2020.csv")

df['MA'] = df['VWAP'].rolling(window=20).mean()