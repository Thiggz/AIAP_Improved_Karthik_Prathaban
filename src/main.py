import pandas as pd
import os
import sqlite3
import yaml
from machine_learning import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocess


with open('config.YAML') as file:
    config = yaml.safe_load(file)

print(config["ordinal"])

path = os.getcwd()
parent = os.path.dirname(path)
path = os.path.join(parent, config["database"])


conn = sqlite3.connect(path)
df = pd.read_sql_query("SELECT * from survive", conn)

data = preprocess.PreProcess(df)
print(data.col_abs)

y = data.preprocessed["Survive"]
x = data.preprocessed.drop("Survive", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["test_size"], random_state=30)

knn_preds = models.knn(x_train, y_train, x_test)
bbpreds = models.balanced_bagging(x_train, y_train, x_test)
rfpreds = models.random_forest(x_train, y_train, x_test)

knn_acc = accuracy_score(y_test, knn_preds)
bb_acc = accuracy_score(y_test, bbpreds)
rf_acc = accuracy_score(y_test, rfpreds)

print("Accuracy: KNN: ", knn_acc, "Balanced Bagging: ", bb_acc, "Random Forest: ", rf_acc)

