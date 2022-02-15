import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# load the data
data = pd.read_csv('C:\\Users\\ehold\\Desktop\\Folders\\Datasets\\melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def missingValues(X_train, X_valid, y_train, y_valid):
    # there are 3 ways to deal with missing values
    # drop columns with missing values - downside is model loses access to alot of potential information
    # imputation - filling in missing values with some number - i.e. the mean of the column - usually leads to a more accurate model than dropping
    # extending imputation - impute the missing values, add a new column to make not of the imputed entries w/ true/false

    # examining data
    # Shape of training data (num_rows, num_columns)
    print(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    approach_to_use = 3

    if approach_to_use == 1:
        # first approach - dropping missing values
        reduced_X_train = X_train.drop(cols_with_missing, axis=1)
        reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

        print("MAE from Approach 1 (Drop columns with missing values):")
        print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
    elif approach_to_use == 2:
        # second approach - imputing missing values with the mean of each column with simple imputer
        my_imputer = SimpleImputer()
        imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
        imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

        # Imputation removed column names; put them back
        imputed_X_train.columns = X_train.columns
        imputed_X_valid.columns = X_valid.columns

        print("MAE from Approach 2 (Imputation):")
        print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
    elif approach_to_use == 3:
        # third approach - imputing missing values with the mean of each column then making a new column to mark the rows imputed
        # Make copy to avoid changing original data (when imputing)
        X_train_plus = X_train.copy()
        X_valid_plus = X_valid.copy()

        # Make new columns indicating what will be imputed
        for col in cols_with_missing:
            X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
            X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

        # Imputation
        my_imputer = SimpleImputer()
        imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
        imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

        # Imputation removed column names; put them back
        imputed_X_train_plus.columns = X_train_plus.columns
        imputed_X_valid_plus.columns = X_valid_plus.columns

        print("MAE from Approach 3 (An Extension to Imputation):")
        print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))



missingValues(X_train, X_valid, y_train, y_valid)

