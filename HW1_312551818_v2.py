import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

features_index = [9]

def extract_feature(dataframe, index_of_feature):
   return dataframe.loc[dataframe['ItemName'] == dataframe.loc[index_of_feature].at["ItemName"], '0':'23']

def convert_to_int_dataframe(dataframe):
    res = dataframe.copy()
    # Replace invalid data with a "NaN" error value
    for i in res.columns:
      res[i] = pd.to_numeric(res[i],errors='coerce')
    return res

def extract_features(dataframe, list_features_index):
    res = []
    features = []
    for i in list_features_index:
        features = convert_to_int_dataframe(extract_feature(dataframe, i)).to_numpy()
        res.append(features.reshape(features.shape[0]*features.shape[1]))
    return np.vstack(res)

def extract_mins_maxs(list):
    res_maxs, res_mins = [],[]
    for i in range(len(list)):
        res_maxs.append(np.nanmax(list[i]))
        res_mins.append(np.nanmin(list[i]))
    return res_maxs, res_mins

def normalize(list):
    res = list.copy()
    maxs, mins = extract_mins_maxs(list)
    for i in range(len(list)):
        for j in range(len(list[0])):
            value = list[i][j]
            if (not np.isnan(value)):
                res[i][j] = (value-mins[i])/(maxs[i]-mins[i])
    return res

y = extract_features(df_train, [9])
min_max_y = extract_mins_maxs(y)
y = normalize(y)
print(y)
# Remove the first value (since we are trying to predict PM2.5 based on previous hour)
y = y[0][1:]
print(y)

X = extract_features(df_train, features_index)
mins_maxs_x = extract_mins_maxs(X)
X = normalize(X)
print(X)
# Remove the last value for each features list (since we are trying to predict PM2.5 based on previous hour)

def offset(list):
    res = []
    for i in range(len(list)):
        res.append(list[i][:-1])
    return np.vstack(res)

X = offset(X)
print(X)

def linear_regression(target,features):
    X = features.copy()
    X = np.concatenate((np.ones((1,len(X[0]))), X),axis=0)
    X = np.transpose(X)
    X = np.nan_to_num(X)
    tX = X.T
    y = np.nan_to_num(target)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(tX,X)), tX), y.T)

# Faut que je dé-"normalize"
print(linear_regression(y,X))