import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pylot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from numpy import asarray
from numpy import savetxt

# Read file
df = pd.read_csv("../fifa_train.csv")
df.head

# Impute missing values
df.isnull().sum()

# Drop unneccessary columns
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