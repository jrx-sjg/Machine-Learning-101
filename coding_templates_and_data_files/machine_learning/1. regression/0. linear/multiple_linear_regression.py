# Multiple Linear Regression
# data preprocessing
# data about 50 companies about their expenses and their profits

# 5 methods of building models
    # 1 All-in (means through all variables )
    # Backward Elimination  ---------- (stepwise regression)
    # Forward Selection ---------- (stepwise regression)
    # Bidirectional Elimination ---------- (stepwise regression)

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:] # it will less or delete One dummy variable i.e from 3 to 2 dummy variables


# Spliting the dataset into the Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, #20%  of whole dataset will be test dataset
                                                    random_state = 0)
# Fitting Multiple Linear REgression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictng the Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
# Backward Elimination
    # 1. Select a significance level to stay in the model (e.g SL = 0.05)
    # 2. Fit the full model with all possible predictors
    # 3. Consider the predictor with the highest P-values. 
        # if P>SL, go to STEP 4, otherwise go to Finish
    # 4. Remove the predictor
    # 5. Fit model without this variable *

#--------------------------------------------------------------#

# 1. Select a significance level to stay in the model (e.g SL = 0.05)
#import statsmodels.formula.api as sm #by course
import statsmodels.regression.linear_model as sm
# X = np.append(arr = X, 
#              values = np.ones((50, 1)).astype(int), #ccolumn of 50 rows and 1 column
#              axis = 1) # this will append column with values 1 at end of X 
# to bring at 1st exchange values
X = np.append(arr = np.ones((50, 1)).astype(int), 
              values = X,
              axis = 1)
#X_opt = X[:, [0,1,2,3,4,5]] #by course
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)

#--------------------------------------------------------------#

# 2. Fit the full model with all possible predictors
# OLS stands for OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#--------------------------------------------------------------#

# 3. Consider the predictor with the highest P-values. 
    #if P>SL, go to STEP 4, otherwise go to Finish
print(regressor_OLS.summary())

#--------------------------------------------------------------#

# 4. Remove the predictor
# the 3rd dummy variable which is at index 2 has highest P value and greater than 0.05
# Remove index 2 column
X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)

#--------------------------------------------------------------#

# 5. Fit model without this variable *
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#--------------------------------------------------------------#

# Repeat step 3 and so on
print(regressor_OLS.summary())
# the 2nd dummy variable which is at index 1 has highest P value and greater than 0.05
# Remove index 1 column
X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

# the administration variable which is at index 2 has highest P value and greater than 0.05
# Remove index 2 column
X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())


# the marketing variable which is at index 2 has highest P value and greater than 0.05
# Remove index 2 column
X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

# the R&D is the only powerful predictor to predict the profit in true sense because its P value is very very small and has greater significance