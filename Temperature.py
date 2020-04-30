import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load
data = pandas.read_csv('data.csv', sep=';', parse_dates=['Time'])
data = data.loc[data['Series'] == 'MB']

data.set_index(pandas.to_datetime(data['Time'], utc=True), inplace=True, drop=True)

data_short = data['2019-08-29 14']

data_short.plot.scatter(x="Time", y="Value")

# ax = data.plot.hist(bins=100, alpha=0.5)

plt.show()

print(data_short)