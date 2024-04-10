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
print(y)
min_max_y = extract_mins_maxs(y)
y = normalize(y)
# Remove the first value (since we are trying to predict PM2.5 based on previous hour)
y = y[0][1:]

X = extract_features(df_train, features_index)
print(X)
mins_maxs_x = extract_mins_maxs(X)
X = normalize(X)
# Remove the last value for each features list (since we are trying to predict PM2.5 based on previous hour)

def offset(list):
    res = []
    for i in range(len(list)):
        res.append(list[i][:-1])
    return np.vstack(res)

X = offset(X)

def linear_regression(target,features):
    X = features.copy()
    X = np.concatenate((np.ones((1,len(X[0]))), X),axis=0)
    X = np.transpose(X)
    X = np.nan_to_num(X)
    tX = X.T
    y = np.nan_to_num(target)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(tX,X)), tX), y.T)

coeffs = linear_regression(y,X)
print(coeffs)

def unnormalize(coeffs,mins_maxs):
    res = coeffs.copy()
    for i in range(len(coeffs)-1):
        res[i] = (coeffs[i]*(mins_maxs[0][0] - mins_maxs[1][0])+mins_maxs[1][0])
    return res

coeffs_normalized = unnormalize(coeffs,mins_maxs_x)
print(coeffs_normalized)

def r2_score(y,X,m,c):
    y_no_nan = np.nan_to_num(y)
    X_no_nan = np.nan_to_num(X)
    ss_t = 0
    ss_r = 0
    for i in range(int(len(X_no_nan))):
        y_pred = c + m * X_no_nan[i]
        ss_t += (y_no_nan[i] - np.mean(y_no_nan)) ** 2
        ss_r += (y_no_nan[i] - y_pred) ** 2
    return 1 - (ss_r/ss_t)

def plot(y_values,X_values,m,c):
    X_values_norm = [i*(mins_maxs_x[0][0] - mins_maxs_x[1][0])+mins_maxs_x[1][0] for i in X_values]
    y_values_norm = [i*(min_max_y[0][0] - min_max_y[1][0])+min_max_y[1][0] for i in y_values]

    # Plot the regression line along with the data points
    x = np.linspace(np.nanmax(X_values_norm), np.nanmin(y_values_norm), 100)
    y = c + m * x

    plt.plot(x, y, color='#58b970', label='Regression Line')
    plt.scatter(X_values_norm, y_values_norm, c='#ef5423', label='data points')

    plt.xlabel('Value of the feature "AMB_TEMP" (H-1)')
    plt.ylabel('Value of PM 2.5')
    plt.legend()
    plt.title('R2: ' + str(r2_score(y_values_norm, X_values_norm, m, c)))
    plt.show()

plot(y,X[0],coeffs_normalized[1],coeffs_normalized[0])

"""
def linear_regression_test(x, y):
    X,Y= np.nan_to_num(x),np.nan_to_num(y)
    mean_x, mean_y = np.mean(X),np.mean(Y)
    m = len(X)
    numer = 0
    denom = 0
    for i in range(m):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    m = numer / denom
    c = mean_y - (m * mean_x)
    return m,c 

print(linear_regression_test(offset(extract_features(df_train, features_index))[0],extract_features(df_train, [9])[0][1:]))
"""