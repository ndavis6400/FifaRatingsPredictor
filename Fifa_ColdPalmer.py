import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pylot as plt
from numpy import asarray
from numpy import savetxt

# Read file
df = pd.read_csv("../fifa_train.csv")
df.head

# Impute missing values
df.isnull().sum()

# Drop unnecessary columns
df.drop(['Photo'], axis=1, inplace = True)
df.drop(['Flag'], axis=1, inplace = True)
df.drop(['Club'], axis=1, inplace = True)
df.drop(['CAM'], axis=1, inplace = True)
df.drop(['CB'], axis=1, inplace = True)
df.drop(['CDM'], axis=1, inplace = True)
df.drop(['CF'], axis=1, inplace = True)
df.drop(['CM'], axis=1, inplace = True)
df.drop(['LAM'], axis=1, inplace = True)
df.drop(['LB'], axis=1, inplace = True)
df.drop(['LCB'], axis=1, inplace = True)
df.drop(['LCM'], axis=1, inplace = True)
df.drop(['LDM'], axis=1, inplace = True)
df.drop(['LF'], axis=1, inplace = True)
df.drop(['LM'], axis=1, inplace = True)
df.drop(['LS'], axis=1, inplace = True)
df.drop(['LW'], axis=1, inplace = True)
df.drop(['LWB'], axis=1, inplace = True)
df.drop(['RAM'], axis=1, inplace = True)
df.drop(['RB'], axis=1, inplace = True)
df.drop(['RCM'], axis=1, inplace = True)
df.drop(['RDM'], axis=1, inplace = True)
df.drop(['RF'], axis=1, inplace = True)
df.drop(['RM'], axis=1, inplace = True)
df.drop(['RS'], axis=1, inplace = True)
df.drop(['RW'], axis=1, inplace = True)
df.drop(['RWB'], axis=1, inplace = True)
df.drop(['ST'], axis=1, inplace = True)

def process_data(df):
    # Removing first 7 characters from the column
    df['Value'] = df['Value'].str[7:]

    # Checking the final character and performaing operations 
    df['Value'] = df['Value'].apply(lambda x: float(x[:-1])* 1000 if x[-1] == 'K' else (float(x[:-1]) *100000 if x[-1] == 'M' else x))

    return df

df = pd.get_dummies(df)

print(df.isnull().sum())

train, test = train_test_split(df, test_size = 0.2)

# Cleaned dataset with unnecessary columns
df.to_csv("../fifa_cleaned_trained.csv")

x_train = train.drop("overall", axis =1)
y_train = train['overall']

x_test = test.drop("overall", axis =1)
y_test = test['overall']

rmse_val = []

for k in range(20):
    k = k+1
    model = neighbors.KNeighborsRegressor(n_neighbors= k)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    error = sqrt(mean_squared_error(y_test, prediction))
    rmse_val.append(error)
    print('RMSE value for k = ' , k , 'is:', error)

curve = pd.DataFrame(rmse_val)
curve.plot()

k = 9
model = neighbors.KNeighborsRegressor(n_neighbors= k)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
evaluation = model.score(x_test, y_test)
print('predictions: \n', prediction, '\n')
print(y_test, '\n')
print('score: ', evaluation, '\n')

test.to_csv('../test.csv')
y_test.to_csv("../y_test.csv")

predictions = pd.DataFrame(prediction, columns = ['overall_prediction'])
predictions.to_csv('../predictions.csv', index = False)

new_test = pd.read_csv('../fifa_test.csv')
submission = pd.read_csv("../fifa_cleaned_trained.csv")
# Still need to join the datasets

# Add joined columns to formula
new_test.drop

# Predicting test set and creating file

predict = model.predict(new_test)
submission['overall'] = predict
submission.to_csv('../fifa_ColdPalmer_Predictions', index = False)