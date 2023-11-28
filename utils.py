from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def preprocess_train(_data, is_remove_outliers):
    data = _data.copy()
    data = data.drop("match_id", axis = 1)
    if is_remove_outliers:
        data = delete_outliers(data)
    data = inner_preprocess(data)
    return split_train(data)

def preprocess_for_result(_data):
     data = _data.copy()
     data = inner_preprocess(data)
     return data

def inner_preprocess(_data):
    data = _data.copy()
    data = delete_not_gold(data)
    data = data.dropna()
    return data

def delete_not_gold(_data : pd.DataFrame):
    data = _data.copy()
    for cat in data.columns:
        if cat != "target":
            if not "gold" in cat:
                data = data.drop(cat, axis = 1)
    return data

def split_train(_data):
    raw_train, raw_test = train_test_split(_data, test_size=0.2, random_state=random_seed)
    Y_train = pd.DataFrame(raw_train['target'])
    X_train = raw_train.drop('target', axis = 1)
    Y_test = pd.DataFrame(raw_test['target'])
    X_test = raw_test.drop('target', axis = 1)
    Y_train["target"] = Y_train['target'].astype(int)
    Y_test["target"] = Y_test['target'].astype(int)
    return X_train,Y_train, X_test, Y_test

def delete_outliers(_data):
    data = _data.copy()
    data = data.where(data["game_time"] < 10000000)
    return data

def write(model):
    writeData = pd.read_csv("DOTA2_TEST_features.csv")
    ids = writeData["match_id"]
    writeData = writeData.drop("match_id", axis = 1)
    writeData = preprocess_for_result(writeData)
    resultWriteData = model.predict(writeData)
    resultWriteData = pd.DataFrame({"match_id" : ids, "radiant_win" : resultWriteData})
    resultWriteData.to_csv("result.csv", index= False)