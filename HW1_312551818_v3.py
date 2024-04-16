import csv
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cost_function(X, y, theta):
    m = y.size
    error = np.matmul(X, theta.T) - y
    cost = 1/(2*m) * np.matmul(error.T, error)
    return cost, error

def gradient_descent(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1/m) * np.matmul(X.T, error))
        cost_array[i] = cost
    return theta, cost_array

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()

# indexs du GOAT : 0,2,4,7,8,9,10,11,12,15,16
# indexs de Noah 0,1,2,3,4,5,6,7,8,10,13,14
features_index = [0,1,2,3,4,5,6,7,8,10,13,14]

def extract_feature(dataframe, index_of_feature):
   return dataframe.loc[dataframe['ItemName'] == dataframe.loc[index_of_feature].at["ItemName"], '0':'23']

def convert_to_int_dataframe(dataframe):
    res = dataframe.copy()
    # Replace invalid data with a "NaN" error value
    for i in res.columns:
      res[i] = pd.to_numeric(res[i],errors='coerce')
    return (res.interpolate(method="polynomial", order=3, limit=None)).to_numpy()

def extract_features(dataframe, list_features_index):
    res = []
    features = []
    for i in list_features_index:
        features = convert_to_int_dataframe(extract_feature(dataframe, i))
        res.append(features.reshape(features.shape[0]*features.shape[1]))
    return np.vstack(res)

df_train = pd.read_csv('train.csv')

y = extract_features(df_train, [9])
# Remove the first value (since we are trying to predict PM2.5 based on previous hour)
y = y[0][1:]
y = np.transpose(y)
y = pd.DataFrame(y, columns=["pm2.5"])
y = y["pm2.5"]

X = extract_features(df_train, features_index)
def offset(list):
    res = []
    for i in range(len(list)):
        res.append(list[i][:-1])
    return np.vstack(res)

X = offset(X)

def interpolate_manuel(arr):
    interpolated_arr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if np.isnan(arr[i][j]):
                # Rechercher les indices précédent et suivant la valeur invalide
                prev_index = j - 1
                next_index = j + 1
                while prev_index >= 0 and np.isnan(arr[i][prev_index]):
                    prev_index -= 1
                while next_index < len(arr) and np.isnan(arr[i][next_index]):
                    next_index += 1

                # Calculer la valeur interpolée en fonction des valeurs voisines valides
                if prev_index >= 0 and next_index < len(arr):
                    interpolated_arr[i][j] = (arr[i][prev_index] + arr[i][next_index]) / 2
                elif prev_index >= 0:
                    interpolated_arr[i][j] = arr[i][prev_index]
                elif next_index < len(arr):
                    interpolated_arr[i][j] = arr[i][next_index]
                else:
                # Si aucune valeur valide n'est trouvée avant ou après, remplacer par 0
                    interpolated_arr[i] = 0
    return interpolated_arr

X = interpolate_manuel(X)
print(X)
X = np.transpose(X)
X = pd.DataFrame(X)

def linear_regression(X_input,y_input):
    X = X_input.copy()
    y = y_input.copy()
    print(X)
    # Normalize our features
    X = (X - X.mean()) / X.std()

    # Add a 1 column to the start to allow vectorized gradient descent
    X = np.c_[np.ones(X.shape[0]), X] 

    # Set hyperparameters
    alpha = 0.1
    iterations = 1000

    # Initialize Theta Values to 0
    theta = np.zeros(X.shape[1])
    initial_cost, _ = cost_function(X, y, theta)

    print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))

    # Run Gradient Descent
    theta, cost_num = gradient_descent(X, y, theta, alpha, iterations)

    # Display cost chart
    plotChart(iterations, cost_num)

    final_cost, _ = cost_function(X, y, theta)

    print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))

    return theta/10

theta = linear_regression(X, y)

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

def extract_last(dataframe, list_features_index):
    res = []
    features = []
    for i in list_features_index:
        features = convert_to_int_dataframe(extract_last_value(dataframe, i))
        res.append(features.reshape(features.shape[0]*features.shape[1]))
    return np.vstack(res)

def extract_last_value(dataframe, index_of_feature):
   return dataframe.loc[dataframe['ItemName'] == dataframe.loc[index_of_feature].at["ItemName"], '8':'8']

df_test = pd.read_csv('test.csv')

test_values = extract_last(df_test,features_index)
test_values = np.concatenate((np.ones((1,len(test_values[0]))), test_values),axis=0)

def compute_pm25(values, c):
    return np.matmul(np.nan_to_num(values),c)

results = compute_pm25(test_values.T,theta)
print(results)

def format_export_answers(missing_pm_values):
    print("Formating and writing answers in csv file...")
    fields = ["index","answer"]
    rows = []
    for i in range(len(missing_pm_values)):
        rows.append(["index_"+str(i), missing_pm_values[i]])

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    with open('answer_' + str(dt_string), 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)
    print("Succesfully formated and wrote answers in a csv file !")

# Format and export the values for export
format_export_answers(results)

#plot(y.to_numpy(),X.to_numpy().T[0],theta[1],theta[0])

