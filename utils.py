from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import torch.nn as nn
import wandb
import torch
random_seed = 42

not_delete_in_heroes = ['_x', '_y', '_level']

def preprocess_train(_data, is_remove_outliers, is_split = True):
    data = _data.copy()
    data = data.drop("match_id", axis = 1)
    if is_remove_outliers:
        data = delete_outliers(data)
    data = inner_preprocess(data)
    if is_split:
        return split_train(data)
    else:
        return data

def preprocess_for_result(_data):
     data = _data.copy()
     data = inner_preprocess(data)
     return data

def inner_preprocess(_data : pd.DataFrame):
    data = _data.copy()
    data = delete_all(data)
    for side in ['r', 'd']:
        
    #print('x')
    data = data.dropna()
    return data

def delete_all(_data : pd.DataFrame):
    data = _data.copy()
    for cat in data.columns:
        if cat != "target":
                data = data.drop(cat, axis = 1)
    return data
def delete_not_gold(_data : pd.DataFrame):
    data = _data.copy()
    for cat in data.columns:
        if cat != "target":
            if not "gold" in cat:
                data = data.drop(cat, axis = 1)
    return data

def delete_not_id(_data : pd.DataFrame):
    data = _data.copy()
    for cat in data.columns:
        if cat != "target":
            if not "id" in cat:
                data = data.drop(cat, axis = 1)
    return data


def delete_not_x_y(_data : pd.DataFrame):
    data = _data.copy()
    for cat in data.columns:
        if cat != "target":
            if not "_x" in cat and not "_y" in cat or "xp" in cat:
                data = data.drop(cat, axis = 1)
    return data

def delete_not_r(_data : pd.DataFrame):
    data = _data.copy()
    for cat in data.columns:
        if cat != "target":
            side = 'r'
            if not 'r' in cat:
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

def write(model, read_from = "DOTA2_TEST_features.csv", drop_index = False):
    writeData = pd.read_csv(read_from)
    if drop_index:
        writeData = writeData.drop("Unnamed: 0", axis = 1)
    ids = writeData["match_id"]
    writeData = writeData.drop("match_id", axis = 1)
    writeData = preprocess_for_result(writeData)
    resultWriteData = model.predict(writeData)
    resultWriteData = pd.DataFrame({"match_id" : ids, "radiant_win" : resultWriteData})
    resultWriteData.to_csv("result.csv", index= False)

def writeNN(net, read_from = "DOTA2_TEST_features.csv", drop_index = False):
    writeData = pd.read_csv(read_from)
    if drop_index:
        writeData = writeData.drop("Unnamed: 0", axis = 1)
    ids = writeData["match_id"]
    writeData = writeData.drop("match_id", axis = 1)
    writeData = preprocess_for_result(writeData)
    dataloader = torch.utils.data.DataLoader(torch.FloatTensor(writeData.values), batch_size=8, shuffle=True,)
    test_preds = np.array([])
    for x_batch in dataloader:
        predictions = net(x_batch)
        predictions = torch.transpose(predictions, 0, 1)[0]
        predictions = predictions.detach().numpy()
        test_preds = np.append(test_preds, predictions)
    resultWriteData = pd.DataFrame({"match_id" : ids, "radiant_win" : test_preds})
    resultWriteData.to_csv("result.csv", index= False)

class MyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        assert len(args) > 0
        len_base = len(args[0])
        
        for i in args:
            assert len(i) == len_base
        
        self.data = args
        
    def __len__(self, ):
        return len(self.data[0])
        
    def __getitem__(self, idx):
        return [t[idx] for t in self.data]