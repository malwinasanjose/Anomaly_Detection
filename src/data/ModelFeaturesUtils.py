import pandas as pd
import numpy as np
import logging


def drop_null_columns(df, threshold=0.01):
    """drop columns with null values more threshold % or higher of the total rows
    :return: dataframe
    """
    df_clean = df.loc[:, df.isna().sum() < df.shape[0]*threshold]
    logging.info(f"{df.shape[1] - df_clean.shape[1]} null columns dropped: {list(df.columns[df.isna().sum() >= df.shape[0]*threshold])}")
    return df_clean


def drop_low_variance_columns(df, threshold=0.001):
    """drop columns with standard deviation below the threshold, 0.001 by default
    :return: dataframe
    """
    try:
        ins_date = df.pop('INSDATE')
        df_clean = df.loc[:, df.std() > threshold]
        df_clean['INSDATE'] = ins_date
    except KeyError:
        df_clean = df.loc[:, df.std() > threshold]
        logging.info(f"{df.shape[1] - df_clean.shape[1]} low variance columns dropped: {list(df.columns[df.std()<=threshold])}")
    return df_clean


def drop_null_rows(df):
    df_clean = df.dropna(how='any', axis=0)
    logging.info(f"{df.shape[0] - df_clean.shape[0]} null rows dropped")
    return df_clean


def drop_timestamp(df):
    try:
        df_clean = df.drop('INSDATE', axis=1)
        logging.info('INSDATE dropped')
    except KeyError:
        df_clean = df.copy()
    return df_clean


def clean_scalar_data(df, null_threshold=0.01, var_threshold=0.001):
    logging.info("cleaning scalar data")
    df_clean = (df.pipe(drop_null_columns, threshold=null_threshold)
                .pipe(drop_low_variance_columns, threshold=var_threshold)
                .pipe(drop_null_rows)
                )

    logging.info(f"{df.shape[1] - df_clean.shape[1]} columns and {df.shape[0] - df_clean.shape[0]} rows dropped")
    return df_clean
