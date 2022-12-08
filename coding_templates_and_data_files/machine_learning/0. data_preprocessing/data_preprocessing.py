# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')

# Outputs the columns Country -> Salary + all it's values
X = dataset.iloc[:, :-1].values

# Outputs the last column + all its values
y = dataset.iloc[:, 3].values

#-------------------------------------------------

# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# We grab only the columns with the missing data
imputer = imputer.fit(X[:, 1:3])

# Replace the missing fields of data with the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])

#-------------------------------------------------

# Import library and classes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X = LabelEncoder()

#Change Country Values in first column to an array<br> of 'label numbers' & adds them to the X
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

# Splits the Country column into 3 separate columns
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Change the Purchase column using LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#-------------------------------------------------

# Spliting the dataset into the Training set & Test set
from sklearn.model_selection import train_test_split

#Make it so that the Test set is 20% and Training set is 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#-------------------------------------------------

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

# Scale the X Training set by fitting it X & then transforming it
# X Train must be done first before X Test
# This is to ensure that they will both be on the same scale
X_train = sc_X.fit_transform(X_train)

# Scale the X Test set by transforming it
# We don't need to fit this because it is already fitted to the X Train
X_test = sc_X.transform(X_test)
