import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def clean_df(df):
    # drop columns containing only NAs
    df_clean = df.dropna(how='all', axis=1)
    print(f'dropped {df.shape[1] - df_clean.shape[1]} columns')
    
    # drop rows with NA values
    df_clean = df_clean.dropna(how='any', axis=0)
    print(f'dropped {df.shape[0] - df_clean.shape[0]} rows')
    
    return df_clean


def plot_time_series(df, columns, index_is_timestamp=False, start_date=None, end_date=None, ma_nr=[], groupby=[], width=10, height=5, anomaly_columns=[], anomaly_values=[]):
    assert isinstance(ma_nr, list) , 'ma_nr should be a list'
    assert isinstance(groupby, list), 'gropuby should be a list'
    assert isinstance(columns, list), 'columns should be a list'
    
    if index_is_timestamp:
        plot_df = df.loc[start_date:end_date]
    else:
        plot_df = df.set_index('INSDATE').loc[start_date:end_date]
    
    # filter specific machine number
    if ma_nr:
        plot_df = plot_df.loc[plot_df['MA_NR'].isin(ma_nr)]
    else:
        pass
    # group by columns
    if groupby:
        plot_df = plot_df.groupby(groupby)
    else:
        pass
    
    n = len(columns)
    
    if anomaly_columns:
        assert len(anomaly_values) == len(anomaly_columns), 'please provide anomaly value for each anomaly column indicator'
        
        m = len(anomaly_columns)
        fig, axs = plt.subplots(n, m, figsize=(width*m, height*n))
        # reformat axs so it can be subset in the event that there's only one row or only one column
        if n==1:
            axs=[axs]
        if m==1:
            axs=[[i] for i in axs]
    
        for col, anomaly in enumerate(anomaly_columns):
            for row, column in enumerate(columns):
                plot_df[column].plot(legend=True, ax=axs[row][col], xlabel='', ylabel=column, alpha=0.5)
                sns.scatterplot(x=plot_df.index[plot_df[anomaly]==anomaly_values[col]], 
                                y=plot_df[column].loc[plot_df[anomaly]==anomaly_values[col]], 
                                color="red", s=10, ax=axs[row][col], label=f'anomaly: {anomaly}', alpha=1)
    else:
        fig, axs = plt.subplots(n, 1, figsize=(width, height*n))
        if n == 1:
            axs = [axs]
        for row, column in enumerate(columns):
                plot_df[column].plot(legend=True, ax=axs[row], xlabel='', ylabel=column, alpha=0.5)
        axs[n-1].set_xlabel(df.index.name)
        
        
def nio_labels(nio_series):
    nio_df = pd.DataFrame(nio_series.astype(str).str.rjust(10,'0').apply(lambda x: [i for i in x] if len(x)==10 else None).apply(pd.Series))
    nio_df.columns = ['1,000,000,000', '100,000,000', '10,000,000', '1,000,000', '100,000', '10,000', '1,000', '100', '10', '1']
    return nio_df