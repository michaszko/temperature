import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy.stats import mode
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, ArtistAnimation

_resample = None
_freq_per_day = None
_data_type = None

pd.set_option('display.max_colwidth', None)


def load_data(input_file):
    # Load data form .csv file

    # MA: eventually, I'm not sure if
    # redefining Python default functions as variables is a good
    # idea... So changed type to data_type
    if _data_type == 'P':  # Pioterk data
        data = pd.read_csv(input_file,
                           sep=',',
                           parse_dates=['time'],
                           index_col='time')

        data = data.drop(columns=['name', 'state', 'host'])

        # Set time as index of the list. After this operation list
        # has only two columns: Time (which is aslo index) and Value
        data.set_index(pd.to_datetime(data.index, unit='ns'),
                       inplace=True,
                       drop=False)

    ###############################################

    elif _data_type == 'D':  # Diablo data
        data = pd.read_csv(input_file,
                           sep=';',
                           parse_dates=['Time'],
                           index_col='Time')

        # Leave only MB
        data = data.loc[data['Series'] == _variable]

        # Set time as index of the list. After this operation list
        # has only two columns: Time (which is aslo index) and Value
        data.set_index(pd.to_datetime(data.index, utc=True, unit="s"),
                       inplace=True,
                       drop=False)
    else:
        print("ERROR: unknown data type specified")
    if data.empty:
        print("no data loaded ")

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


def filter(df, fmode):
    '''  
        Resolution of thermometer sometimes casuing date to oscilate.
        To properly analize data one has to get rid of those
        oscialtions  using some low pass filter.
    '''
    # I quickly implemented moving average: window paramter means how
    # much data has to be taken under consideration
    if fmode == 1:
        # 1. Mode (most repeated value)
        return df.rolling(window=5).apply(lambda x: mode(x)[0])
    elif fmode == 2:
        # 2. Median
        return df.rolling(window=10, center=True).median()
    elif fmode == 3:
        # 3. Similar to rolling() but argument is time not number of
        #    data points
        return df.resample(_resample).median()
    else:
        print("ERROR: unknown filter mode specified")


def decomp(df):
    '''
    Returns domcompositon of the data into
    1. trend
    2. sesonal changes 
    3. residusal
    '''
    return seasonal_decompose(df, model='additive', period=_freq_per_day)


def split(df):
    '''
    Devides data into days or weeks or months
    '''
    if _data_split == 'M':
        # Divide into months
        df.index = [df.index.day, df.index.time, df.index.month]

    if _data_split == 'W':
        # Divide into weeks
        df.index = [df.index.weekday, df.index.time, df.index.week]

    if _data_split == 'D':
        # Divide into days
        df.index = [df.index.time, df.index.date]

    return df.unstack()

def normalize(df):
    '''
    Return dataframe devided into days from which everyone start at value 0
    '''

    df.index = [df.index.date, df.index.time]

    df = df.rename_axis(index=['Date', 'Time'])

    midnight = pd.Timestamp(2017, 1, 1, 0).time()

    ind = df.index.get_level_values('Date').unique()

    dfs = pd.DataFrame()

    for i in ind:
        dfs[i] = df[i] - df.loc[(i, midnight)]

    return dfs

def plot_stacked_days(df, data_split = 'D'):

    if data_split == 'D':
        mrange = df.index.values
    else:
        mrange = range(len(df.index))

    # standard deviation of time series
    mstd = df.std(axis=1)
    # average of time series
    mavg = df.mean(axis=1)

    ax = df.plot(legend=0, alpha=0.3)

    # Plot averge
    ax = mavg.interpolate().plot(linewidth=2, linestyle="-", color="blue")

    # Plot std
    ax.fill_between(mrange, mavg - mstd, mavg + mstd, color='b', alpha=0.2)

    ax.figure.autofmt_xdate()
    ax.set_title("Stacked " + data_split)
    ax.set_ylabel("Temperature  [$\\degree$ C]")
    ax.set_xlabel(None)

    plt.show()

    # Average day in the sample
    ax = df.mean(axis=1).plot()

    # Plot std
    plt.fill_between(mrange,
                     mavg - mstd,
                     mavg + mstd,
                     color='b',
                     alpha=0.2)

    ax.set_title("Average " + data_split)
    ax.set_ylabel("Temperature  [$\\degree$ C]")
    ax.set_xlabel(None)

    plt.show()

#########################################################


def main(input_file, data_type, data_split, rstime, fmode, variable,
         decomposition, sdays, phase):
    # TOOD by MA: move all global variables setting to a separate
    # function
    global _resample
    global _data_type
    global _data_split
    global _variable
    _resample = rstime
    _data_type = data_type
    _data_split = data_split
    _variable = variable

    # Load data
    data = load_data(input_file)

    # Average points with the same date
    data = data.groupby(data.index).mean()

    # Calculate the time span of the data in seconds
    time_span = pd.Timedelta(
        data.tail(1).index.values[0] - data.head(1).index.values[0]).seconds

    # Calculate frequency of data points
    data_freq = pd.Timedelta(
        data.tail(1).index.values[0] - data.tail(2).index.values[0]).seconds

    # Normalize frequacy to days and also set it global
    global _freq_per_day
    _freq_per_day = round(24 * 60 * 60 / data_freq)

    # Change from DataFrame to Series
    df = data.squeeze()

    # One can look at smaller pieces of data -- you short it as
    # following
    # Winter
    # df = df['2018-11-01':'2019-04-01']
    # Summer
    # df = df['2019-05-01':'2019-09-01']
    # df = df['2019-05-01':'2019-05-31']

    #################################################################

    # Comparison of smoothed and raw data
    #
    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].set_ylabel("Temperature  [$\\degree$ C]")
    axes[1].set_ylabel("Temperature/second \
    [${}^{\\degree}\\mathrm{C}/_\\mathrm{sec}$]")

    axes[0].set_title("Data")
    axes[1].set_title("Derivative")

    df.plot(label="Raw", legend=True, ax=axes[0])
    filter(df, fmode).plot(label="Filtered", legend=True, ax=axes[0])
    # plt.show()

    # Playing with derivatives
    #

    derivative(df).plot(label="Raw", legend=True, ax=axes[1])
    derivative(filter(df, fmode)).plot(label="Der. of filtered", legend=True)
    derivative(filter(df, fmode)).rolling(window=10).sum().plot(
        label="Sum of der. of filtered", legend=True)
    plt.show()

    # Playing with autocorrelation -- corellation between data and
    # shifted data. For some data it is visible that data is
    # correlated after shifting ~24h

    # pd.plotting.autocorrelation_plot(df,
    #                              label="Autocorrelation")

    # plt.show()

    # Playing with decomposition
    # getting enabled with flag
    if decomposition:
        decomp(df).plot()
        plt.show()

    # Stacked days
    #
    if sdays:

        only_open = False
        _normalize = True

        if only_open and data_split == 'D':

            dfs = filter(df, fmode)

            x = derivative(df).rolling(window=10).sum()
            y = x[x < -3] 
            # I assume that doors are open when derivative drops at
            # least 3-4 deegree
            open_doors_days = y.groupby(y.index.date).sum().index

            dfs.index = [dfs.index.date, dfs.index.time]

            dfs = dfs.rename_axis(index=['Date', 'Time'])

            # Days when doors were open
            # dfs = dfs[dfs.index.isin(open_doors_days, level='Date')]
            # Days when doors were closed
            dsf = dfs[~dfs.index.isin(open_doors_days, level='Date')]

            dfs = dfs.unstack(level=0)

            #########################################################

            plot_stacked_days(df)

        if _normalize and data_split == 'D':

            dfs = normalize(filter(df, fmode))

            plot_stacked_days(dfs)


        plot_stacked_days(split(filter(df, fmode)), data_split)

        #############################################################

    if phase:

        dff = pd.DataFrame()

        # Phase diagram of avarage quantities
        if int(phase) == 0:
            dff["avg"] = split(filter(df, fmode)).mean(axis=0)
            dff["std"] = split(filter(df, fmode)).std(axis=0)
            dff["max"] = split(filter(df, fmode)).max(axis=0)
            dff["min"] = split(filter(df, fmode)).min(axis=0)
            dff["spread"] = dff["max"] - dff["min"]

            xplot = "avg"
            yplot = "std"

            if data_split == 'D':
                ax = dff.plot.scatter(xplot,
                                      yplot,
                                      c=dff.index.month,
                                      colormap='viridis')
            else:
                ax = dff.plot.scatter(xplot,
                                      yplot,
                                      c=dff.index,
                                      colormap='viridis')

        # Phase diagram of all data points
        elif int(phase) == 1:
            _window = 10

            dff["norm"] = filter(df, fmode)
            dff["der"] = derivative(filter(
                df, fmode)).rolling(window=_window).sum()
            dff["der2"] = derivative(
                derivative(filter(df, fmode)).rolling(
                    window=_window).sum()).rolling(window=_window).sum()

            xplot = "norm"
            yplot = "der"

            ax = dff.plot.scatter(xplot,
                                  yplot,
                                  c=dff.index.month,
                                  colormap='viridis')

        # ax.plot(np.linspace(-15,15,100),0.75*np.linspace(-15,15,100))

        # ax.set_title("Average temp. of " + data_split + " with its std")
        ax.set_xlabel(xplot)
        ax.set_ylabel(yplot)

        # plt.show()

        #############################################################

        # Histograms

        fig, ax = plt.subplots(1, 1)

        x = dff[xplot].to_numpy()
        y = dff[yplot].to_numpy()

        # Remove nan and zeros
        y = y[~np.isnan(x)]
        x = x[~np.isnan(x)]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        # norm = norm[der != 0]
        # der2 = der2[der != 0]
        # der = der[der != 0]

        plt.hist2d(
            x,
            y,
            bins=20,
            #range=[[np.min(nor[np.nonzero(nor)]),np.max(nor[np.nonzero(nor)])],[der.min(),der.max()]],
            cmap=plt.cm.jet,
            norm=LogNorm())

        ax.set_xlabel(xplot)
        ax.set_ylabel(yplot)

        plt.colorbar(ax=ax)
        plt.show()

        #############################################################

        # win = 5
        # fig, ax = plt.subplots(win,win)
        # fig.set_tight_layout(True)

        # for i in range(win):
        #     for j in range(1, win+1):
        #         dff = pd.DataFrame()

        #         dff["norm"] = filter(df, fmode)
        #         dff["der"] = derivative(filter(df, fmode)).rolling(window=win*i+j).sum()
        #         dff["der2"] = derivative(derivative(filter(df, fmode)).rolling(window=win*i+j).sum()).rolling(window=win*i+j).sum()

        #         xplot = "norm"
        #         yplot = "der"

        #         ax[i,j-1].scatter(dff[xplot], dff[yplot], c=dff.index.month, cmap='viridis')

        #         # ax[i,j].set_xlabel(xplot)
        #         # ax[i,j].set_ylabel(yplot)
        #         ax[i,j-1].set_title(str(win*i+j))

        # plt.show()


# ______________________________________________________________________________________________________________________________

if __name__ == "__main__":

    import optparse
    parser = optparse.OptionParser(
        usage="%prog [options]..",
        description="Create plots of the temperature fluctuations ")

    parser.add_option('-i',
                      '--input-file',
                      dest="input_file",
                      default="mb.csv",
                      help='Data file')

    parser.add_option(
        '-t',
        '--data-type',
        dest="data_type",
        default="P",
        help='Data  type. Put "P" for Pioterk data and "D" for Diablo')

    parser.add_option(
        '--data-split',
        dest="data_split",
        default="D",
        help=
        'Data  spliting range. Put "D" for days,"W" for weeks and  "M" for months'
    )

    parser.add_option(
        '--filter',
        dest="fmode",
        default=3,
        help=
        'Data filter mode. Choose from 1,2 or 3. Where 1 corresponds to  mode (most repeated value), 2 Median, 3 Similar to rolling() but argument is time not number of data points  '
    )

    parser.add_option(
        '--resample-time',
        dest="rstime",
        default="10min",
        help=
        'In case of choosing 3 filter you need ro set resample time in the following format: 30min (default)'
    )

    parser.add_option(
        '-v',
        '--variable',
        dest="variable",
        default="MB",
        help=
        'Choose the source for the remperature: sensor on teh CPU or MB(default)'
    )

    parser.add_option(
        '-d',
        '--decomposition',
        dest="decomposition",
        default=False,
        action="store_true",
        help='Allows to enable decomposition of the signal. Default: False')

    parser.add_option('-s',
                      '--stacked-days',
                      dest="sdays",
                      default=False,
                      action="store_true",
                      help='Enables option to plot stacked days')

    parser.add_option('-p',
                      '--phase-diagram',
                      dest="phase",
                      default=False,
                      help='Enables option to plot phase diagrams')

    o, other = parser.parse_args()

    main(o.input_file, o.data_type, o.data_split, o.rstime, o.fmode,
         o.variable, o.decomposition, o.sdays, o.phase)

    if other:
        print("Warning - ignored arguments: ", other)
