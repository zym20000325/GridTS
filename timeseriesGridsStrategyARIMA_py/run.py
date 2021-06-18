from timeseriesGridStrategyARIMA import timeseriesGridStrategyARIMA
import pandas as pd

data = pd.read_csv("data/example10.csv")
grids = timeseriesGridStrategyARIMA(data)
print(grids)

data = pd.read_csv("data/example11.csv")
grids = timeseriesGridStrategyARIMA(data)
print(grids)

data = pd.read_csv("data/example12.csv")
grids = timeseriesGridStrategyARIMA(data)
print(grids)