from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import pandas as pd
import os
import logging


def get_outlier_scores(df, file_name, machine_settings, n_estimators, max_samples, contamination=0.01,
                       model='IsolationForest'):
    scores_dir = f"../../data/outlier_scores/{machine_settings['ma_nr']}"
    subdir = f"{machine_settings['wsg_id']}_{machine_settings['wzd_id']}_{machine_settings['st_id']}_{machine_settings['at_id']}_{machine_settings['start_date']}_{machine_settings['end_date']}"
    path = f"{scores_dir}/{subdir}"
    file = f"{path}/{file_name}"

    if os.path.exists(file):
        scores = pd.read_pickle(file)
        logging.info(f"loaded scores from {file}")
    else:
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

        if model == 'IsolationForest':
            outlier_detector = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                               contamination=contamination)
            outlier_detector.fit(df)
            labels = outlier_detector.predict(df)
            scores = pd.DataFrame({'scores': outlier_detector.score_samples(df), 'labels': labels},
                                  index=df.index)
            scores.to_pickle(file)
            logging.info(f"saved scores in {file}")
        else:
            raise Exception('Only IsolationForest implemented at the moment')

    return scores


def get_outlier_labels(scores, threshold=-0.5):
    labels = scores['scores'].apply(lambda x: -1 if x < threshold else 1).to_frame(name='labels')
    return labels


def get_percent_threshold(scores, percentile=0.01):
    threshold = scores['scores'].quantile(percentile)
    return threshold


def get_feature_importances(df, labels, file_name, machine_settings):
    f_dir = f"../../data/feature_importances/{machine_settings['ma_nr']}"
    subdir = f"{machine_settings['wsg_id']}_{machine_settings['wzd_id']}_{machine_settings['st_id']}_{machine_settings['at_id']}_{machine_settings['start_date']}_{machine_settings['end_date']}"
    path = f"{f_dir}/{subdir}"
    file = f"{path}/{file_name}"

    if os.path.exists(file):
        feature_importances = pd.read_pickle(file)
        logging.info(f"loaded feature_importances from {file}")
    else:
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

        # fit a tree regressor
        class_weights = labels.value_counts(normalize=True).to_dict()
        classifier = DecisionTreeClassifier(class_weight=class_weights)
        classifier.fit(df, labels)
        # get permutation feature importances

        feature_importances = pd.DataFrame(
            {'feature': df.columns, 'mean_importance': classifier.feature_importances_}).sort_values(
            'mean_importance').reset_index()
        #save results
        feature_importances.to_pickle(file)
        logging.info(f'feature importances saved: {file}')
    return feature_importances
