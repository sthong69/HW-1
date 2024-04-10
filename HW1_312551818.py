import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Formatting data to only keep PM2.5 values
pm_values = df_train.loc[df_train['ItemName'] == df_train.loc[9].at["ItemName"], '0':'23']
pm_test_values = df_test.loc[df_test['ItemName'] == df_test.loc[9].at["ItemName"], '0':'8']

def convert_to_int_dataframe(dataframe):
    # Replace invalid data with a "NaN" error value
    for i in dataframe.columns:
      dataframe[i] = pd.to_numeric(dataframe[i],errors='coerce')
    # Convert the dataframe to a numpy array
    return dataframe.to_numpy()

# Converting dataframe to int-only numpy array
pm_values = convert_to_int_dataframe(pm_values)
pm_test_values = convert_to_int_dataframe(pm_test_values)

def extract_pm_values(int_dataframe):
    X, Y = [], []
    for i in range(int_dataframe.shape[0]):
        for j in range(int_dataframe.shape[1] - 1):
         # Checks if the values are not NaN
            if (not np.isnan(int_dataframe[i][j]) and not np.isnan(int_dataframe[i][j+1])):
                X.append(int_dataframe[i][j])
                Y.append(int_dataframe[i][j+1])
    return X,Y

# Extract and separate consecutive values of n pm2.5 values from n+1 hour pm2.5 values
X_train, Y_train = extract_pm_values(pm_values)
X_test, Y_test = extract_pm_values(pm_test_values)

def linear_regression(X, Y):
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

# Compute the values of m and c with the training set (given y = mx + c)
m, c = linear_regression(X_train,Y_train)
print("Value for m with test values = " + str(m))
print("Value for c with test values = " + str(c))

# Plot the regression line along with the data points
x = np.linspace(np.max(X_train), np.min(Y_train), 100)
y = c + m * x

plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X_train, Y_train, c='#ef5423', label='data points')

plt.xlabel('PM2.5 of current hour')
plt.ylabel('PM2.5 of next hour')
plt.legend()
plt.show()

def RMSE(real_values, predicted_values):
    res = 0
    for i in range(len(real_values)):
        res += (predicted_values[i]-real_values[i])**2
    return (res/len(real_values))**(1/2)

# Compute the performance on our training data based on RMSE
print(RMSE(Y_train,[m * i + c for i in X_train]))
print(RMSE(Y_test,[m * i + c for i in X_test]))

# Compute the PM2.5 missing values
print(pm_test_values)
missing_pm_values = []
for i in range(len(pm_test_values)):
    missing_pm_values.append(m*pm_test_values[i][-1] + c)
print(len(missing_pm_values))

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
format_export_answers(missing_pm_values)

"""
# Compute R-squared value to assess the goodness of our model. 

ss_t = 0 #total sum of squares
ss_r = 0 #total sum of square of residuals

for i in range(int(len(X))): # val_count represents the no.of input x values
  y_pred = c + m * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print("R2 = " + str(r2))
"""

