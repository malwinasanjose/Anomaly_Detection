import logging

import json
import os
import pandas as pd

import panel as pn

from make_dataset import MachineData
from get_outliers import get_outlier_scores, get_outlier_labels

logging.basicConfig(format='%(asctime)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# specify location of db credentials file
if os.path.exists('.config'):
    with open('.config', 'r') as file:
        config_path = json.load(file)
else:
    config_path = input("where is the config file")
    with open('.config', 'w') as file:
        file.write(json.dumps(config_path))

# get user inputs for machine number and input configuration settings
machine = MachineData(config_path)

machine_numbers = machine.get_machine_numbers()

machine_selection = pn.widgets.Select(name='Please select a machine number:',
                                      options={f"{row[0]} - {row[1]} records": row[0]
                                               for row in machine_numbers.values.tolist()}
                                      )


@pn.depends(machine_selection)
def get_configs():
    machine.settings['MA_NR'] = int(machine_selection.value)
    return machine.get_machine_configurations()


config_selection = pn.widgets.Select(name='Please select a configuration setting:',
                                     options=get_configs().index.tolist())


@pn.depends(config_selection)
def assign_configuration_settings():
    machine_configurations = get_configs(machine_selection)
    for col in machine_configurations.columns:
        try:
            machine.settings[col] = machine_configurations[col][config_selection.value].item()
        except AttributeError:
            machine.settings[col] = str(machine_configurations[col][config_selection.value].date())


model_df = machine.get_model_data()
spek_plot_df = machine.get_plotting_spectral_data()

model_scores = get_outlier_scores(model_df, machine.settings, model='IsolationForest', n_estimators=500,
##                                  max_samples=0.8)
model_labels = get_outlier_labels(model_scores, threshold=-0.5)

dash = pn.Column(machine_selection, config_selection)
dash.servable()



