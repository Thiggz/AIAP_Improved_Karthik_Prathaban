import pandas as pd
import os
import sqlite3
import yaml
from machine_learning import models
from sklearn.model_selection import train_test_split
from preprocessing import preprocess
from evaluation import model_eval

## Read config file

with open('config.YAML') as file:
    config = yaml.safe_load(file)

## Open the database

path = os.getcwd()
parent = os.path.dirname(path)
path = os.path.join(parent, config["database"])

conn = sqlite3.connect(path)
df = pd.read_sql_query("SELECT * from survive", conn)

## Perform preprocessing

print('DataFrame open, Performing Preprocessing\n')
data = preprocess.PreProcess(df)

y = data.preprocessed[config['dependent']]
x = data.preprocessed.drop(config['dependent'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["test_size"], random_state=30)

## Train classifiers

print('Training KNN, Balanced Bagging and Random Forest Classifiers')
knn_trained = models.knn(x_train, y_train)
bb_trained = models.balanced_bagging(x_train, y_train)
rf_trained = models.random_forest(x_train, y_train)
print('Training complete\n')

## Test classifiers and print default evaluation metrics

if config["test_size"] > 0:

    print('Test set available, testing models\n')

    knn_preds = knn_trained.predict(x_test)
    bb_preds = bb_trained.predict(x_test)
    rf_preds = rf_trained.predict(x_test)

    predictions = {'K-Nearest Neighbours': knn_preds,
                   'Balanced Bagging': bb_preds,
                   'Random Forest': rf_preds}

    for key in predictions:
        model_eval.default_metrics(y_test, predictions[key], key)
