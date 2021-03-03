from make_dataset import get_df_from_db, get_model_spectral_data, get_plotting_spectral_data, merge_scalar_spectral
from ModelFeaturesUtils import clean_scalar_data
from get_outliers import get_outlier_scores, get_outlier_labels, get_percent_threshold, get_feature_importances
import json
import panel as pn
import param
import os
import yaml
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Span, Label

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pn.extension('plotly')

# connect to database
import sqlalchemy as db

import logging

logging.basicConfig(format='%(asctime)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

MODEL_SPEK_SENSORS = ['SPEK_R1_1',
                      'SPEK_R2_1',
                      'SPEK_R3_1',
                      'SPEK_RP_1',
                      'SPEK_R1_2',
                      'SPEK_R2_2',
                      'SPEK_R3_2',
                      'SPEK_RP_2',
                      'SPEK_L1',
                      'SPEK_L2',
                      'SPEK_L3',
                      'SPEK_LP']
PLOT_SPEK_SENSORS = ['FFT_RED_SPEK_L1',
                     'FFT_RED_SPEK_L2',
                     'FFT_RED_SPEK_L3',
                     'FFT_RED_SPEK_LP',
                     'FFT_RED_SPEK_R1',
                     'FFT_RED_SPEK_R2',
                     'FFT_RED_SPEK_R3',
                     'FFT_RED_SPEK_RP',
                     'FFT_RED_FREQ_SPEK_RP',
                     'FFT_RED_FREQ_SPEK_R1',
                     'FFT_RED_FREQ_SPEK_R2',
                     'FFT_RED_FREQ_SPEK_R3',
                     'FFT_RED_FREQ_SPEK_L1',
                     'FFT_RED_FREQ_SPEK_L2',
                     'FFT_RED_FREQ_SPEK_L3',
                     'FFT_RED_FREQ_SPEK_LP']
SCALAR_SENSORS = ['IMAX_R',
                  'IMAX_L',
                  'IRMS_R',
                  'IRMS_L',
                  'IINT_R',
                  'IINT_L',
                  'IMAX_R_abHUB2',
                  'IRMS_R_abHUB2',
                  'IINT_R_abHUB2',
                  'POSI_X',
                  'POSI_Y',
                  'POSI_Y_L',
                  'POSI_Z',
                  'ORD1_R',
                  'ORD2_R',
                  'ORDz1_I_R',
                  'ORDz2_I_R',
                  'ORDz3_I_R',
                  'ORDsum_R',
                  'ORD1_L',
                  'ORD2_L',
                  'ORDz1_I_L',
                  'ORDz2_I_L',
                  'ORDz3_I_L',
                  'ORDsum_L',
                  'ORDb_aR',
                  'ORDz1_aR',
                  'ORDz2_aR',
                  'ORDz3_aR',
                  'ORDsum_aR',
                  'ORDAmax_aR',
                  'ORDFmax_aR',
                  'ORDb_aL',
                  'ORDz1_aL',
                  'ORDz2_aL',
                  'ORDz3_aL',
                  'ORDsum_aL',
                  'ORDAmax_aL',
                  'ORDFmax_aL',
                  'AXIS_C_ACTIVE',
                  'FORCE_FAKTOR',
                  'APM_SCHL_Aktiv',
                  'SCHLEIFZEIT',
                  'IRMS_V',
                  'IMAX_Xanfahrt',
                  'IINT_V',
                  'VORSTUFE_AKTIV',
                  'IMAX_V',
                  'IINTOFF_R']

# specify location of db credentials file
if os.path.exists('.config'):
    with open('.config', 'r') as file:
        config_path = json.load(file)
else:
    config_path = input("where is the config file")
    with open('.config', 'w') as file:
        file.write(json.dumps(config_path))

config = yaml.safe_load(open(config_path))
engine = db.create_engine(
    '{DB_DRIVER}://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?driver={DB_ODBC_DRIVER}'.format(**config)
)


class MachineSelection(param.Parameterized):
    _machine_numbers = get_df_from_db(file='sql/get_machine_numbers.sql', engine=engine)
    _machine_numbers = {f"{row[0]} - {row[1]} records": row[0]
                        for row in _machine_numbers.values.tolist()}
    _machine_numbers.update({" ": -1})

    _machine_configurations = None
    _settings = {'ma_nr': None,
                 'wsg_id': None,
                 'wzd_id': None,
                 'st_id': None,
                 'at_id': None,
                 'start_date': None,
                 'end_date': None,
                 'warm': 0,
                 'teach_active': 0,
                 'ready': 1}

    scalar_df = None
    spek_plot_dict = None
    scores_df = None
    labels_df = None
    outliers = None
    cds = None
    model_df = None
    feature_importances = None

    plot_sensor_objects = []

    p_ma_nr = param.ObjectSelector(default=-1, objects=_machine_numbers, label='Select a machine number:')
    p_config = param.ObjectSelector(default=-1, objects={" ": -1}, label='Select an operating configuration:')
    p_get_data = param.Action(lambda x: x.param.trigger('p_get_data'), label='Retrieve Data')
    p_scalar_sensor = param.ObjectSelector(objects=[], label='Select a sensor:', precedence=-1)
    p_spectral_sensor = param.ObjectSelector(objects=[], label='Select a sensor:', precedence=-1)
    p_cutoff_score = param.Number(default=-0.500, bounds=(-1, 0), step=0.01, label='Specify an anomaly cutoff score:',
                                  precedence=-1)
    p_percent_outliers = param.Number(default=0.01, bounds=(0.01, 0.1), step=0.01, label='Specify anomaly percentage:',
                                      precedence=-1)
    p_bins = param.Integer(default=20, bounds=(5, 100), step=5, label='Number of histogram bins', precedence=-1)

    # hidden parameters
    # spectral data processing settings
    p_rerun_processing = param.Boolean(default=False, precedence=-1)
    p_agg = param.Boolean(default=False, precedence=-1)
    p_smoothing = param.String(default='savgol', precedence=-1)
    p_polyorder = param.Integer(default=1, precedence=-1)
    p_window_length = param.Integer(default=3, precedence=-1)
    p_pca_components = param.Number(default=0.8, precedence=-1)
    p_contamination = param.Number(default=0.01, precedence=-1)

    # outlier detection model parameters
    p_outlier_detection_model = param.String(default='IsolationForest', precedence=-1)
    p_n_estimators = param.Integer(default=1000, precedence=-1)
    p_max_samples = param.Number(default=1.0, precedence=-1)

    @param.depends('p_ma_nr', watch=True)
    def _update_config(self):
        self._settings['ma_nr'] = self.p_ma_nr
        available_configurations = get_df_from_db(file='sql/get_machine_configurations.sql', engine=engine,
                                                  **self._settings)
        self._machine_configurations = available_configurations
        available_configurations = {
            f"{row.start_date} to {row.end_date}: {row.records} records (WSG_ID={row.WSG_ID}, WZD_ID={row.WZD_ID}, ST_ID={row.ST_ID}, AT_ID={row.AT_ID})": row.Index
            for row in available_configurations.itertuples()}
        config_dict = {" ": -1}
        config_dict.update(available_configurations)

        for key, value in config_dict.items():
            self.param.p_config.names[key] = value
        self.param['p_config'].objects = config_dict.values()

    @param.depends('p_config', watch=True)
    def _update_settings(self):
        self._settings['ma_nr'] = self.p_ma_nr
        conf = self._machine_configurations.iloc[self.p_config]
        conf.index = conf.index.str.lower()
        conf = conf.to_dict()
        self._settings.update(conf)

    @param.depends('p_get_data', watch=True)
    def _retrieve_data(self):
        # get input data
        scalar_df = clean_scalar_data(get_df_from_db(file='sql/get_sensor_data.sql', engine=engine, index='PRIMARY',
                                                     sensors=', '.join(SCALAR_SENSORS + ['INSDATE']),
                                                     **self._settings)).sort_values('INSDATE', axis=0)

        spek_df = get_model_spectral_data(file='sql/get_sensor_data.sql', engine=engine, index='PRIMARY',
                                          sensors=', '.join(MODEL_SPEK_SENSORS), **self._settings,
                                          agg=self.p_agg,
                                          smoothing=self.p_smoothing,
                                          polyorder=self.p_polyorder,
                                          window_length=self.p_window_length,
                                          pca_components=self.p_pca_components,
                                          rerun_processing=self.p_rerun_processing, )

        spek_plot_dict = get_plotting_spectral_data(file='sql/get_sensor_data.sql', engine=engine, index='PRIMARY',
                                                    sensors=', '.join(PLOT_SPEK_SENSORS), **self._settings)
        model_df = merge_scalar_spectral(scalar_df, model_spek_df=spek_df)
        # update class attributes
        self.scalar_df = scalar_df
        self.spek_plot_dict = spek_plot_dict
        self.model_df = model_df
        self.cds = ColumnDataSource(scalar_df)
        # update dropdown menu for plotting sensors based on what's available in the data
        self.param['p_scalar_sensor'].objects = [" "] + scalar_df.drop('INSDATE', axis=1).columns.tolist()
        self.param['p_spectral_sensor'].objects = [" "] + list(spek_plot_dict.keys())

    @param.depends('model_df', 'p_outlier_detection_model', 'p_n_estimators', 'p_max_samples', watch=True)
    def _get_outlier_scores(self):
        # scores will change if any of the input data or model parameter changes; configuration is saved in file name
        scores_file = f"outlier_scores_{self.p_outlier_detection_model}_{self.p_n_estimators}_{str(self.p_max_samples).replace('.', '-')}_{self.p_agg}_{self.p_smoothing}_{self.p_polyorder}_{self.p_window_length}_{str(self.p_pca_components).replace('.', '-')}_{self.p_contamination}.pkl"
        if self.model_df is not None:
            # drop some columns from model
            drop_cols = ['POSI_X', 'POSI_Y', 'POSI_Y_L', 'POSI_Z', 'AXIS_C_ACTIVE', 'FORCE_FAKTOR', 'APM_SCHL_Aktiv',
                         'SCHLEIFZEIT', 'VORSTUFE_AKTIV']
            reduced_model_df = self.model_df.drop([i for i in self.model_df.columns if i in drop_cols], axis=1)
            # run outlier detection and get scores and labels
            scores_df = get_outlier_scores(reduced_model_df, scores_file, self._settings,
                                           model=self.p_outlier_detection_model,
                                           n_estimators=self.p_n_estimators,
                                           max_samples=self.p_max_samples,
                                           contamination=self.p_contamination)
            # update class attributes
            self.scores_df = scores_df
            self.param.p_cutoff_score.precedence = 0
            self.param.p_percent_outliers.precedence = 0
            self.param.p_bins.precedence = 0
        else:
            pass

    @param.depends('scores_df', 'p_cutoff_score', watch=True)
    def _get_outlier_labels(self):
        if self.scores_df is not None:
            labels_df = get_outlier_labels(self.scores_df, threshold=self.p_cutoff_score)
            outliers = labels_df[['labels']].loc[labels_df['labels'] == -1].join(self.scalar_df, how='inner')
            self.labels_df = labels_df
            self.outliers = outliers
        else:
            pass

    @param.depends('scores_df', watch=True)
    def _get_feature_importances(self):
        if self.scores_df is not None and self.model_df is not None:
            file = f"feature_importances_{self.p_outlier_detection_model}_{self.p_n_estimators}_{str(self.p_max_samples).replace('.', '-')}_{self.p_agg}_{self.p_smoothing}_{self.p_polyorder}_{self.p_window_length}_{str(self.p_pca_components).replace('.', '-')}_{self.p_contamination}.pkl"
            feature_importances = get_feature_importances(self.model_df, self.scores_df['labels'], file, self._settings)
            self.feature_importances = feature_importances
        else:
            pass

    @param.depends('feature_importances', watch=True)
    def _update_plotting_params(self):
        if self.feature_importances is not None:
            columns = self.feature_importances['feature']
            spek_columns = [i[0:7] for i in columns if 'SPEK' in i]
            scalar_columns = [i for i in columns if 'SPEK' not in i]
            top_spek = spek_columns[-1]
            top_scalar = scalar_columns[-1]
            self.param['p_scalar_sensor'].precedence = 0
            #            self.p_scalar_sensor = top_scalar
            self.param['p_spectral_sensor'].precedence = 0

    #            self.p_spectral_sensor = top_spek

    @param.depends('p_percent_outliers', watch=True)
    def _update_p_cutoff_score(self):
        threshold = get_percent_threshold(self.scores_df, self.p_percent_outliers)
        self.p_cutoff_score = threshold

    @param.depends('p_scalar_sensor', 'outliers')
    def scalar_plots(self):
        if self.outliers is not None and self.p_scalar_sensor is not None and self.p_scalar_sensor != " ":
            p = figure(x_axis_type="datetime", width=800, height=300, x_axis_label='Timestamp')
            p.line('INSDATE', self.p_scalar_sensor, source=self.cds, line_width=1)
            p.circle(self.outliers['INSDATE'], self.outliers[self.p_scalar_sensor], color="red", size=1,
                     legend_label='Anomalies')
            return p
        else:
            pass

    @param.depends('p_spectral_sensor', 'outliers')
    def spectral_plots(self):
        if self.outliers is not None and self.p_spectral_sensor is not None and self.p_spectral_sensor != " ":
            fig = go.Figure(layout=go.Layout(width=800, height=400, margin={'b': 0, 'l': 0, 't': 0, 'r': 30},
                                             scene_aspectmode='manual',
                                             scene_aspectratio=dict(x=0.5, y=1, z=0.5),
                                             legend=dict(yanchor='top',
                                                         y=0.99,
                                                         xanchor='left',
                                                         x=0.01)
                                             ))
            fig.update_scenes(xaxis_title_text='Frequency', yaxis_title_text='Gear ID', zaxis_title_text='Magnitude')
            plot_dat = self.spek_plot_dict[self.p_spectral_sensor]
            plot_outliers = plot_dat.join(self.outliers['labels'], how='inner', on='PRIMARY')
            fig.add_trace(go.Scatter3d(name='', x=plot_dat['Frequency'], y=plot_dat['PRIMARY'], z=plot_dat['Magnitude'],
                                       mode='markers',
                                       opacity=0.8,
                                       marker=dict(
                                           color=plot_dat['Magnitude'],
                                           size=1.5,
                                           colorscale='YlGnBu')
                                       ),
                          )
            fig.add_trace(
                go.Scatter3d(name='Anomalies', x=plot_outliers['Frequency'], y=plot_outliers['PRIMARY'],
                             z=plot_outliers['Magnitude'],
                             mode='markers',
                             # mode='lines',
                             opacity=1,
                             marker=dict(
                                 color='red',
                                 size=1, )
                             # line=dict(
                             #    color='red',
                             #    width=2)
                             )
            )
            return fig
        else:
            pass

    @param.depends('feature_importances')
    def plot_feature_importances(self):
        # plot importances for the 30 most important features
        if self.feature_importances is not None:
            f_import_small = self.feature_importances.iloc[-30:, :]

            p = figure(y_range=f_import_small['feature'], width=400, height=800, title='Feature Importances')
            p.hbar(y=f_import_small['feature'], right=f_import_small['mean_importance'], left=0, height=0.2)

            return p
        else:
            pass

    def box_plots(self):
        pass

    @param.depends('p_bins', 'outliers')
    def score_histograms(self):
        if self.scores_df is not None and self.outliers is not None:
            p = figure(title='Anomaly Scores', width=300, height=300)
            hist, edges = np.histogram(self.scores_df['scores'], bins=self.p_bins)

            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                   fill_color='navy', line_color='white', alpha=0.5)
            vline = Span(location=self.p_cutoff_score, dimension='height', line_color='red', line_width=3,
                         line_dash='dashed')
            text = f"{self.outliers.shape[0]} anomalies ({round(100 * self.outliers.shape[0] / self.scores_df.shape[0], 2)} %)"
            x = np.min(edges) * 0.98
            y = np.max(hist) * 0.9
            label = Label(text=text, x=x, y=y, )
            p.add_layout(vline)
            p.add_layout(label)
            return p
        else:
            pass

machine = MachineSelection()

gspec = pn.GridSpec(sizing_mode='stretch_both', height_policy='max', width_policy='max')

gspec[0, :] = pn.Column(pn.pane.Markdown("#  Anomaly Detection Dashboard", style={'font-family': 'Verdana', 'color':'#F1F1F1'},
        background="#1A2425",
        sizing_mode='stretch_width',
        height=50,
        margin=(0,0,0,20)),
        background="#1A2425")

gspec[1:11,0] = pn.Column(
    pn.Column(
    pn.Spacer(background='#EAEBEB',
             height=3,
             sizing_mode='stretch_width'),
    machine.param['p_ma_nr'],
    machine.param['p_config'],
    machine.param['p_get_data'],
    machine.param['p_cutoff_score'],
    machine.param['p_percent_outliers'],
    pn.Spacer(background='#EAEBEB',
              height=10),
    machine.score_histograms,
    machine.param['p_bins'],
    pn.Spacer(background='#EAEBEB',
              sizing_mode='stretch_height'),
    background='#EAEBEB',
    margin=(0,10,0,20),
    width_policy='fit'
    ),
    background='#EAEBEB',
    #margin = (30,0,0,0),
    max_width=350,
    sizing_mode='stretch_both',
    width_policy='fit'
)
gspec[1:11, 1:3] = pn.Column(
            pn.Column(machine.param['p_scalar_sensor'], machine.scalar_plots),
            pn.Column(machine.param['p_spectral_sensor'], machine.spectral_plots),
            width_policy='fit',
            margin=(5,10,0,20)
)
gspec[1:11, 3:4] = pn.Column(
    machine.plot_feature_importances,
    width_policy='max',
    margin=(5,10,0,20))
gspec.servable()