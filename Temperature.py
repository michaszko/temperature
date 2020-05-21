import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy.stats import mode
from statsmodels.tsa.seasonal import seasonal_decompose


def load_data(name='mb.csv', type='P'):
    # Load data form .csv file
    if type == 'P':  # Pioterk data
        data = pd.read_csv(name,
                           sep=',',
                           parse_dates=['time'],
                           index_col='time')

        data = data.drop(columns=['name', 'state', 'host'])

        # Set time as index of the list. After this operation list has only two columns: Time (which is aslo index) and Value
        data.set_index(pd.to_datetime(data.index, unit='ns'),
                       inplace=True,
                       drop=False)

    ###############################################

    if type == 'D':  # Diablo data
        data = pd.read_csv(name,
                           sep=';',
                           parse_dates=['Time'],
                           index_col='Time')

        # Leave only MB
        data = data.loc[data['Series'] == 'MB']

        # Set time as index of the list. After this operation list has only two columns: Time (which is aslo index) and Value
        data.set_index(pd.to_datetime(data['Time'], utc=True, unit="s"),
                       inplace=True,
                       drop=False)

    return data

def plot_data(df):
    '''
        Plotting data
    '''
    df.plot()
    plt.show()


def derivative(df):
    '''
    Calculating derivative of given series of data
    '''
    return pd.Series(np.gradient(df.to_numpy()), df.index, name='slope')


def filter(df, time='30min'):
    '''  
        Resolution of thermometer sometimes casuing date to oscilate.
        To properly analize data one has to get rid of those
        oscialtions  using some low pass filter.
    '''
    # I quickly implemented moving average: window paramter means how
    # much data has to be taken under consideration
    #
    # 1. Mode (most repeated value)
    # return df.rolling(window=5).apply(lambda x: mode(x)[0])]
    #
    # 2. Median
    # return df.rolling(window=10, center=True).median()
    #
    # 3. Similar to rolling() but argument is time not number of data
    #    points
    return df.resample(time).median()


def decomp(df):
    '''
    Returns domcompositon of the data into
    1. trend
    2. sesonal changes 
    3. residusal
    '''
    return seasonal_decompose(df, model='additive', period=freq_per_day)


def split(df, time='D'):
    '''
    Devides data into days or weeks or months
    '''
    if time == 'M':
        # Divide into months
        df.index = [df.index.day, df.index.time, df.index.month]

    if time == 'W':
        # Divide into weeks
        df.index = [df.index.dayofweek, df.index.time, df.index.week]

    if time == 'D':
        # Divide into days
        df.index = [df.index.time, df.index.date]

    return df.unstack()


#####################################################################
# Load data
data =load_data()

# Average points with the same date
data = data.groupby(data.index).mean()

# Calculate the time span of the data in seconds
time_span = pd.Timedelta(
    data.tail(1).index.values[0] - data.head(1).index.values[0]).seconds

# Calculate frequency of data points
data_freq = pd.Timedelta(
    data.tail(1).index.values[0] - data.tail(2).index.values[0]).seconds

# Normalize frequacy to days
freq_per_day = round(24 * 60 * 60 / data_freq)

# Change from DataFrame to Series
df = data.squeeze()

# One can look at smaller pieces of data -- you short it as following
df = df['2019-05-01':'2019-05-30']

#####################################################################

# Comparison of smoothed and raw data
#
fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].set_ylabel("Temperature  [$\\degree$ C]")
axes[1].set_ylabel("Temperature/second \
    [${}^{\\degree}\\mathrm{C}/_\\mathrm{sec}$]")

axes[0].set_title("Data")
axes[1].set_title("Derivative")

df.plot(label="Raw", legend=True, ax=axes[0])
filter(df).plot(label="Filtered", legend=True, ax=axes[0])
# plt.show()

# Playing with derivatives
#
derivative(df).plot(label="Raw", legend=True, ax=axes[1])
derivative(filter(df)).plot(label="Der. of filtered", legend=True)
derivative(filter(df)).rolling(window=10).sum().plot(
    label="Sum of der. of filtered", legend=True)
plt.show()

# Playing with autocorrelation -- corellation between data and
# shifted data. For some data it is visible that data is correlated
# after shifting ~24h

# pd.plotting.autocorrelation_plot(df,
#                                  label="Autocorrelation")

# plt.show()

# Playing with decomposition
#
decomp(df).plot()

plt.show()

# Stacked days
#
ax = split(filter(data)).plot(legend=0)

ax = split(filter(data)).mean(axis=1).plot(linewidth=5,
                                           linestyle=":",
                                           color="red")

ax.figure.autofmt_xdate()
ax.set_title("Stacked days")
ax.set_ylabel("Temperature  [$\\degree$ C]")
ax.set_xlabel(None)

plt.show()

# Average day in the sample
#
ax = split(filter(data)).mean(axis=1).plot()

ax.set_title("Average day")
ax.set_ylabel("Temperature  [$\\degree$ C]")
ax.set_xlabel(None)

plt.show()
