import os
import logging

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from scipy.signal import savgol_filter, medfilt

from read_famos import load_channels


def try_load_channels(x):
    if x is not None:
        try:
            return load_channels(x)
        except ValueError:
            return None
    else:
        return None


def load_series_spectrum_df(series_dict_channels):
    """
    Takes a series of dictionaries generated by pd.Series.apply(load_channels)
    and returns a dataframe with the frequencies expanded as columns.
    If the frequencies are not identically overlapping across rows, the resulting
    set of columns will the the union of all the different frequency sets, where
    rows not containing a given frequency will be NaN
    """
    dict_df = {}
    for i, dict_channels in series_dict_channels.items():
        if dict_channels:
            for key, value_dict in dict_channels.items():
                n_rows = len(value_dict['value_y'])
                x_values = np.array(value_dict['delta_x']).dot(np.arange(n_rows))
                for j, freq in enumerate(x_values):
                    try:
                        dict_df[freq][i] = value_dict['value_y'][j]
                    except KeyError:
                        dict_df[freq] = {i: value_dict['value_y'][j]}
        else:
            pass
    return pd.DataFrame.from_dict(dict_df)


def extract_spectral_data_from_df(df):
    """
    takes a dataframe where each columns is a spectral sensor. Expands each columns into a dataframe and returns a
    dictionary of dataframes
    :param df: dataframe of binary format spectral data
    :return: dictionary of dataframes of expanded spectral data
    """
    spek_dict = {}
    for col in df.columns:
        temp_df = load_series_spectrum_df(df[col].apply(try_load_channels))
        if temp_df.shape == (0, 0):
            logging.info(f'{col} contains no data')
            pass
        elif amp_cut_off(temp_df, amp_threshold=0.09):
            # if all values in the dataframe are below amp_threshold consider the sensor to be mostly noise and discard
            logging.info(f'{col} dropped as noise below default amp_threshold of 0.09')
            pass
        else:
            spek_dict[col] = temp_df

    return spek_dict


def write_spectral_data_to_file(spek_dict_or_df, path):
    if isinstance(spek_dict_or_df, dict):
        for key, df in spek_dict_or_df.items():
            df.to_pickle(f"{path}/{key}.pkl")
    elif isinstance(spek_dict_or_df, pd.DataFrame):
        spek_dict_or_df.to_pickle(path)


def read_spectral_data_from_files(path):
    """
    reads all files in a given directory and stores them into a dictionary of dataframes
    :param path: directory containing extracted spectral data files
    :return: dictionary of dataframes, where the keys are the spectral sensor names
    """
    spek_dict = {}
    for file in os.listdir(path):
        # strip file extension because dict key is used in the columns names downstream
        filename = os.path.splitext(file)[0]
        spek_dict[filename] = pd.read_pickle(f"{path}/{file}")
    return spek_dict


def bin_frequencies(df, how='max', bin_size=5, n_bins=None):
    """
    bins spectral data frequencies to the specified bin size or number of bins
    :param df: dataframe of spectral data from single sensor
    :param how: how to aggregate the intensities for each bins. Any numpy aggregate function, default is max
    :param bin_size: size of frequency bins, default is 5. Overriden by n_bins if specified
    :param n_bins: number of bins of equal size to return. Overrides bin_size. Default is None
    :return: dataframe with same number or rows but reduced number of columns
    """
    df = df.T.reset_index()
    df['index'] = df['index'].astype('float')
    if n_bins:
        f_min = df['index'].min()
        f_max = df['index'].max()
        bin_size = (f_max - f_min) // n_bins
        df['freq_bin'] = (df['index'] // bin_size) * bin_size
    else:
        df['freq_bin'] = (df['index'] // bin_size) * bin_size
    df = df.groupby('freq_bin').agg(how).drop('index', axis=1).T
    return df


def process_spectral_data(spek_dict, pca_components=0.8, agg=True, smoothing='bin', bin_size=5, polyorder=1,
                          window_length=3):
    """
    for each dataframe in a dictionary,
        1. smooths the data by binning, or applying savgol or medfilt filters (from scipy)
        2. apply PCA and extract pca_components number of components, default 80% cumulative explained variance
        3. calculate summary statistics (mean, std, skew, kurtosis, argmax, max) if agg=True
    append resulting columns from each dataframe in the dictionary as columns to a combined dataframe
    and return combined dataframe
    :param spek_dict: dictionary of dataframes containing extracted spectral data
    :param pca_components: number of pca_components to keep
    :param agg: boolean, whether to include summary statistics, default True
    :param smoothing: bin, savgol, or medfilt
    :param kwargs: parameters to pass to smoothing functions
    :return: dataframes of spectral features extracted from each of the dataframes in the spek_dict dictionary
    """

    spek_df = pd.DataFrame()

    for key, df in spek_dict.items():
        # smooth
        if smoothing == 'bin':
            temp = bin_frequencies(df, bin_size=bin_size)
        elif smoothing == 'savgol':
            temp = df.apply(savgol_filter, polyorder=polyorder, window_length=window_length,
                            axis=1, result_type='expand')
        elif smoothing == 'medfilt':
            temp = df.apply(medfilt, axis=1, result_type='expand')
        else:
            temp = df.copy()
        # apply PCA
        temp = pd.DataFrame(PCA(n_components=pca_components).fit_transform(temp), index=df.index)
        temp.columns = [f'{key}_{col}' for col in temp.columns]
        spek_df = spek_df.join(temp, how='outer')
        logging.info(f"{key}: {temp.shape[1]} PCA components")
        # summary statistics
        if agg is True:
            temp = df.agg(['mean', 'std', 'skew', 'kurtosis', 'argmax', 'max'], axis=1)
            temp.columns = [f'{key}_{col}' for col in temp.columns]
            spek_df = spek_df.join(temp, how='outer')
        else:
            pass
    return spek_df


def amp_cut_off(dataframe, amp_threshold=0.09):
    """
    return true if all data is below amp_threshold
    """
    return (np.abs(dataframe.values) < amp_threshold).all()

