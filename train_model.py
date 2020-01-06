# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import json
import pathlib
from pathlib import Path


@click.command()
@click.argument('test_filepath', type=click.Path(exists=True))
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('output_file_directory', type=click.Path())
def main(test_filepath, train_filepath, output_file_directory):
    logger = logging.getLogger(__name__)
    logger.info('Reading data files')
    df_train = pd.read_csv(pathlib.Path(train_filepath), delimiter=";")
    df_test = pd.read_csv(pathlib.Path(test_filepath), delimiter=";")
    logger.info("Training data size = {0}, testing data size = {1}".format(df_train.size, df_test.size))

    logger.info('Constructing pipeline')
    numeric_features = [
        'age',
        'balance',
        'day',
        'campaign',
        'pdays',
        'previous',
    ]
    categorical_features = [
        'job',
        'marital',
        'education',
        'default',
        'housing',
        'loan',
        'contact',
        'month',
        'campaign',
        'pdays',
        'previous',
    ]
    numeric_transformer_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor_pipe = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_pipe, numeric_features),
            ('cat', categorical_transformer_pipe, categorical_features)])
    X_train = df_train.drop('target', axis=1)
    y_train = df_train['target']

    X_test = df_test.drop('target', axis=1)
    y_test = df_test['target']
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor_pipe),
        ('classifier', RandomForestClassifier(n_jobs=-1, n_estimators=100))])

    logger.info('Fitting model')
    clf.fit(X_train, y_train)

    metrics_test = get_metrics(X_test, y_test, clf)
    metrics_train = get_metrics(X_train, y_train, clf)
    metrics_dict = {
        'metrics_test': metrics_test,
        'metrics_train': metrics_train
    }

    pathlib.Path(output_file_directory).mkdir(parents=True, exist_ok=True)
    with open(output_file_directory + '/metrics.json', 'w') as outfile:
        json.dump(metrics_dict, outfile, indent=4)

    param_grid = {
        'classifier__n_estimators': [10, 30, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30]
    }

    logger.info('Calculating grid search')
    grid_search = GridSearchCV(clf, param_grid, cv=5, iid=False, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    params_res = {
        'classifier__n_estimators': grid_search.best_params_['classifier__n_estimators'],
        'classifier__max_depth': grid_search.best_params_['classifier__max_depth']
    }
    with open(output_file_directory + '/best_params.json', 'w') as outfile:
        json.dump(params_res, outfile, indent=4)

    converted_dict = dict()
    for key in grid_search.cv_results_.keys():
        val = grid_search.cv_results_[key]
        converted_val = val
        if isinstance(val, np.ndarray):
            converted_val = val.tolist()
        converted_dict[key] = converted_val

    with open(output_file_directory + '/cv_results.json', 'w') as outfile:
        json.dump(converted_dict, outfile, indent=4)

    logger.info('Finished')


def get_metrics(x, y, clf):
    res = dict()
    tlf = clf.predict(x)
    res['model_accuracy'] = metrics.accuracy_score(y, tlf)
    res['model_precision'] = metrics.precision_score(y, tlf)
    res['model_recall'] = metrics.recall_score(y, tlf)
    res['model_F1'] = metrics.f1_score(y, tlf)
    res['model_AuROC'] = metrics.roc_auc_score(y, tlf)
    return res


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
