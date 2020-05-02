import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack
from scipy import signal


def plot_data(df):
    '''
        Plotting data
    '''
    df.plot()
    plt.show()


def calculate_derivative(df):
    '''
    Calculating derivative of given series of data
    '''
    return pd.Series(np.gradient(df.to_numpy()), df.index, name='slope')


def filter(df):
    '''  
        Resolution of thermometer sometimes casuing date to oscilate.
        To properly analize data one has to get rid of those
        oscialtions  using some low pass filter.
    '''
    # I quickly implemented moving average
    return df.rolling(window=10).mean()


#####################################################################

# Load data form .csv file
data = pd.read_csv('data.csv', sep=';', parse_dates=['Time'])

# Leave only MB
data = data.loc[data['Series'] == 'MB']

# Set time as index of the list. After this operation list has only two columns: Time (which is aslo index) and Value
data.set_index(pd.to_datetime(data['Time'], utc=True, unit="s"),
               inplace=True,
               drop=False)

# Calculate the time span of the data in seconds
time_span = pd.Timedelta(
    data.tail(1).index.values[0] - data.head(1).index.values[0]).seconds

# Average points with the same date
data = data.groupby(data.index).mean()

# Change from DataFrame to Series
df = data.squeeze()

# One can look at smaller pieces of data -- you short it as following
df_s = df['2019-08-29 14']

#####################################################################

# Comparison of smoothed and raw data
df.plot()
df.rolling(window=8).mean().plot()
plt.show()

# plot_data(df)

# low_filter(data)

# print(data)