import csv
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def aberration_correction(data):
    res = data.copy()
    mean, var = np.mean(data, axis=1), np.var(data, axis=1)
    for i in range(len(data)):
        for j in range(len(data[0])):
            test = (data[i][j]-mean[i])/var[i]
            if test>10:
                res[i][j]=mean[i]
    return res

def extract_feature(dataframe, index_of_feature):
   return dataframe.loc[dataframe['ItemName'] == dataframe.loc[index_of_feature].at["ItemName"], '0':'23']

def convert_to_int_dataframe(dataframe):
    res = dataframe.copy()
    # Replace invalid data with a "NaN" error value
    for i in res.columns:
      res[i] = pd.to_numeric(res[i],errors='coerce')
    return (res.interpolate(method="polynomial", order=2, limit=None)).to_numpy()

def extract_features(dataframe, list_features_index):
    res = []
    features = []
    for i in list_features_index:
        features = convert_to_int_dataframe(extract_feature(dataframe, i))
        res.append(features.reshape(features.shape[0]*features.shape[1]))
    return np.vstack(res)

def manual_interpolation(arr):
    res = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if np.isnan(arr[i][j]):
                # We check each value iteratively before/after the error
                previous_index = j - 1
                next_index = j + 1
                while previous_index >= 0 and np.isnan(arr[i][previous_index]):
                    previous_index -= 1
                while next_index < len(arr) and np.isnan(arr[i][next_index]):
                    next_index += 1

                # We estimate the value based on the values found before/after (just a simple mean here)
                if previous_index >= 0 and next_index < len(arr):
                    res[i][j] = (arr[i][previous_index] + arr[i][next_index]) / 2
                elif previous_index >= 0:
                    res[i][j] = arr[i][previous_index]
                elif next_index < len(arr):
                    res[i][j] = arr[i][previous_index]
                else:
                # If we can't find values on either sides, we just replace the error with a 0
                    res[i] = 0
    return res

def search_target_index(list):
    for i in range(len(list)):
        if list[i] == 9:
            return i

def window_sliding_train(data, target_column_index, window_size, limit):
    X, y = [], []

    if limit == 0:
        for i in range(12):
            for j in range(window_size + 480*i, 480*(i+1)):
                window_X = []
                for k in range(len(data)):
                    window_X.append(data[k][j - window_size:j])
                X.append(window_X)
                y.append(data[target_column_index][j])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0],-1)
        return X, y
    
    else:
        day = limit//24
        months = day//20

        for i in range(months):
            for j in range(window_size + 480*i, 480*(i+1)):
                window_X = []
                for k in range(len(data)):
                    window_X.append(data[k][j - window_size:j])
                X.append(window_X)
                y.append(data[target_column_index][j])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0],-1)
        return X, y

def linear_regression(target, features, lamb):
    X = features.copy()
    X = np.transpose(X)
    tX = X.T
    y = np.nan_to_num(target)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(tX,X) + lamb * np.identity(len(X[0]))), tX), y.T)

def compute_pm25(values, c):
    return np.matmul(np.nan_to_num(values),c)

def RMSE(real_values, predicted_values):
    res = 0
    for i in range(len(real_values)):
        res += (predicted_values[i]-real_values[i])**2
    return (res/len(real_values))**(1/2)

"""
def plot(y_values,X_values,m,c):
    # Plot the regression line along with the data points
    x = np.linspace(np.nanmax(X_values), np.nanmin(y_values), 100)
    y = c + m * x

    plt.plot(x, y, color='#58b970', label='Regression Line')
    plt.scatter(X_values, y_values, c='#ef5423', label='data points')

    plt.xlabel('Value of the feature (H-1)')
    plt.ylabel('Value of PM 2.5')
    plt.legend()
    plt.show()
"""

def extract_features_test(dataframe, list_features_index):
    res = []
    features = []
    for i in list_features_index:
        features = convert_to_int_dataframe(extract_feature_test(dataframe, i))
        res.append(features.reshape(features.shape[0]*features.shape[1]))
    return np.vstack(res)

def extract_feature_test(dataframe, index_of_feature):
   return dataframe.loc[dataframe['ItemName'] == dataframe.loc[index_of_feature].at["ItemName"], '0':'8']

def window_sliding_test(data, target_column_index, window_size):
    X, y = [], []
    
    num_windows = len(data[0]) // window_size 
    end_index = 0

    for i in range(num_windows):
        window_X = []
        for j in range(len(data)):
            start_index = i * window_size
            end_index = (i + 1) * window_size
            window_X.append(data[j][start_index:end_index])
        X.append(window_X)
        y.append(data[target_column_index][start_index])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0],-1)
    return X, y

def format_export_answers(missing_pm_values, RMSE_v, list_features):
    res = input("Do you want to output the answers with RMSE = "+str(RMSE_v)+" ? (Y/N)")
    if res == "N":
        return
    elif res == "Y":
        print("Formating and writing answers in csv file...")
        fields = ["index","answer"]
        rows = []
        for i in range(len(missing_pm_values)):
            rows.append(["index_"+str(i), missing_pm_values[i]])

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        with open('answer_' + str(dt_string) + "_" +str(RMSE_v)+"_"+str(list_features)+".csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)
        print("Succesfully formated and wrote answers in a csv file !")
    else:
        format_export_answers(missing_pm_values, RMSE_v, list_features)

def main(features, limit, lamb):
    features_index = features
    nb_of_days_of_features = 9

    if limit>12*20*24:
        print("Limit is too high! Going limitless...") 
        limit = 0

    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    data_train = extract_features(df_train, features_index)
    if limit != 0:
        data_train = limit_data(data_train, limit)
    data_train = aberration_correction(data_train)
    data_train = manual_interpolation(data_train) 

    X_train, y_train = window_sliding_train(data_train, search_target_index(features_index), nb_of_days_of_features, limit)

    X_train = X_train.T
    X_train = np.concatenate((np.ones((1,len(X_train[0]))), X_train),axis=0)
    X_train = X_train.T

    coeffs = linear_regression(y_train,X_train.T, lamb)

    RMSE_value = RMSE(y_train, compute_pm25(X_train, coeffs))
    print("RMSE = " + str(RMSE_value))

    data_test = extract_features_test(df_test, features_index)
    data_test = aberration_correction(data_test)
    data_test = manual_interpolation(data_test)

    X_test, y_test = window_sliding_test(data_test, search_target_index(features_index), nb_of_days_of_features)
    X_test = X_test.T
    X_test = np.concatenate((np.ones((1,len(X_test[0]))), X_test),axis=0)
    X_test = X_test.T

    results = compute_pm25(X_test, coeffs)
    print(results)

    # Format and export the values for export
    format_export_answers(results, RMSE_value, features_index)

def checkpmcorrelation():
    df_train = pd.read_csv('train.csv')
    features = extract_features(df_train,[_ for _ in range(17)])
    features_names = ["AMB_TEMP", "CH4", "CO", "NHMC", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5", "RAINFALL", "RH", "SO2", "THC", "WD_HR", "WIND_DIRECTION", "WIND_SPEED", "WS_HR"]

    for i in range(len(features)):
        plt.subplot(3,6,i+1)
        plt.scatter(features[i],features[9])
        plt.title(features_names[i])
    plt.show()

def correlation_matrix():
    df_train = pd.read_csv('train.csv')
    features = extract_features(df_train,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17])
    
    df = pd.DataFrame(features.T, columns=['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR'])
    correlation_matrix = df.corr()

    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()

    # Create a heatmap using imshow
    cmap = plt.get_cmap('coolwarm')  # Choose a colormap
    cax = ax.imshow(correlation_matrix, interpolation='nearest', cmap=cmap)

    # Add a color bar on the right
    fig.colorbar(cax)

    # Set x and y ticks to feature names
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.columns)
    plt.xticks(rotation=45, ha='right')

    # Add the correlation values as text on each cell
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black")

    ax.set_title('Correlation Matrix')
    plt.show()

def limit_data(data, limit):
    res = []
    for i in range(len(data)):
        res.append(data[i][:limit])
    return res

def check_aberration():
    df_train = pd.read_csv('train.csv')
    features = extract_features(df_train, [3,9])
    plt.scatter(features[1],features[0])
    plt.title("CH4 values without aberration correction")
    plt.show()

    features_corrected = aberration_correction(features)
    plt.scatter(features_corrected[1], features_corrected[0])
    plt.title("CH4 values with aberration correction")
    plt.show()

if __name__ == "__main__":
    # features_index = [0,2,4,7,8,9,10,11,12,15,16]
    # indexs v1 : 0,2,4,7,8,9,10,11,12,15,16
    # indexs v2 0,1,2,3,4,5,6,7,8,9,10,13,14
    # v3 2,3,4,6,8,9,16
    features_index =  [0,2,4,7,8,9,10,11,12,15,16]
    #checkpmcorrelation()
    #correlation_matrix()
    main(features_index, limit = 0, lamb = 100000)
    #check_aberration()
    
