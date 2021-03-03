# -*- coding: utf-8 -*-

import logging


# basic packages
import os
# import numpy as np
import pandas as pd


# custom functions
from SpectralUtils import read_spectral_data_from_files, process_spectral_data, write_spectral_data_to_file, \
    extract_spectral_data_from_df
from ModelFeaturesUtils import clean_scalar_data, drop_timestamp


def get_df_from_db(file, engine, index=None, **kwargs):
    sql = open(file).read().format(**kwargs)
    with engine.connect() as conn:
        df = pd.read_sql(sql=sql, con=conn)
        if index is not None:
            df.set_index(index, inplace=True)
    return df


def get_model_spectral_data(file, engine, index=None, rerun_processing=False, agg=True, smoothing='bin', bin_size=5,
                            pca_components=0.8, polyorder=1, window_length=3, **kwargs):
    logging.info('getting spectral data')
    # to avoid re-running the extraction of spectral data:
    # create a folder with the machine name and specific configuration settings
    # create files for each sensor and processed and combined data
    raw_spek_dir = "../../data/raw/spectral_data/{ma_nr}".format(**kwargs)
    processed_spek_dir = "../../data/processed/spectral_data/{ma_nr}".format(**kwargs)
    spek_subdir = "{wsg_id}_{wzd_id}_{st_id}_{at_id}_{start_date}_{end_date}".format(**kwargs)

    # directory if if doesn't exist
    if not os.path.exists(f"{processed_spek_dir}/{spek_subdir}"):
        os.makedirs(f"{processed_spek_dir}/{spek_subdir}")
    else:
        pass

    path = f"{processed_spek_dir}/{spek_subdir}/spek_processed_{agg}_{str(pca_components).replace('.', '')}_{smoothing}_{bin_size}_{window_length}_{polyorder}.pkl"

    # if processed data file exists and rerun_processing is False, read processed data and return it,
    # otherwise run processing steps
    if os.path.exists(path) and rerun_processing is False:
        spek_df = pd.read_pickle(path)
        logging.info(f'{path} file found')
    else:
        # if extracted spectral data exists, read it, otherwise query the db
        if os.path.exists(f"{raw_spek_dir}/{spek_subdir}"):
            spek_dict = read_spectral_data_from_files(f"{raw_spek_dir}/{spek_subdir}")
            logging.info(f'extracted raw spectral data found: {list(spek_dict.keys())}')
        else:
            # query for spectral columns
            raw_spek_df = get_df_from_db(file=file, engine=engine, index=index, **kwargs)
            # extract spectral data and save files in raw_spek_dir/spek_subdir
            spek_dict = extract_spectral_data_from_df(raw_spek_df)
            if len(spek_dict) == 0:
                pass
            else:
                os.makedirs(f"{raw_spek_dir}/{spek_subdir}")
                write_spectral_data_to_file(spek_dict, path=f"{raw_spek_dir}/{spek_subdir}")
                logging.info(f"spectral data extracted and saved: {list(spek_dict.keys())}")

        if len(spek_dict) == 0:
            logging.info('No spectral data available')
            spek_df = pd.DataFrame()
        else:
            spek_df = process_spectral_data(spek_dict, agg=agg, smoothing=smoothing, bin_size=bin_size,
                                            pca_components=pca_components, polyorder=polyorder,
                                            window_length=window_length)

            # save processed data
            write_spectral_data_to_file(spek_df, path=path)
            logging.info(f"processed spectral data saved: {path}")

    return spek_df


def get_plotting_spectral_data(file, engine, index, **kwargs):
    """
    get FFT reduced spectra for plotting
    for each sensor, the frequencies of peaks are contained in the fields with a naming convention FFT_RED_FREQ_sensor
    the corresponding magnitudes are contained in fields with a naming convention FFT_RED_sensor
    :return: dictionary of long format dataframes to use in plotly 3d plotting
    """
    raw_spek_dir = "../../data/raw/plotting_spectral_data/{ma_nr}".format(**kwargs)
    spek_subdir = "{wsg_id}_{wzd_id}_{st_id}_{at_id}_{start_date}_{end_date}".format(**kwargs)
    plot_dict = {}

    if os.path.exists(f"{raw_spek_dir}/{spek_subdir}"):
        plot_dict = read_spectral_data_from_files(f"{raw_spek_dir}/{spek_subdir}")
        logging.info(f'extracted plotting spectral data found: {list(plot_dict.keys())}')
    else:
        raw_spek_df = get_df_from_db(file=file, engine=engine, index=index, **kwargs)
        # get frequency and magnitude column names in separate variables
        spek_sensors = raw_spek_df.columns
        # extract magnitude dataframes
        plot_spek = [i for i in spek_sensors if 'FREQ' not in i]
        spek_dict = extract_spectral_data_from_df(raw_spek_df[plot_spek])
        # extract frequencies dataframes
        plot_spek_freq = [i for i in spek_sensors if 'FREQ' in i]
        spek_freq_dict = extract_spectral_data_from_df(raw_spek_df[plot_spek_freq])
        if any([len(spek_freq_dict) == 0, len(spek_dict) == 0]):
            pass
        else:
            for key, df in spek_freq_dict.items():
                # stack and join frequency and magnitude dataframes
                freq_df = df.stack().to_frame(name='Frequency')
                spek_df = spek_dict.get(key.replace('FREQ_', ''), pd.DataFrame()).stack().to_frame(name='Magnitude')
                plot_df = freq_df.join(spek_df).reset_index().drop('level_1', axis=1).rename({'level_0': 'PRIMARY'},
                                                                                             axis=1)
                plot_dict[key.replace('FFT_RED_FREQ_', '')] = plot_df
            # save files
            os.makedirs(f"{raw_spek_dir}/{spek_subdir}")
            write_spectral_data_to_file(plot_dict, path=f"{raw_spek_dir}/{spek_subdir}")
            logging.info(f"spectral data extracted and saved: {list(plot_dict.keys())}")

    return plot_dict


def merge_scalar_spectral(scalar_df, model_spek_df):
    scalar_df = drop_timestamp(scalar_df)
    model_data = scalar_df.join(model_spek_df, how='inner')
    logging.info(f"model dataframe shape: {model_data.shape}")
    return model_data
