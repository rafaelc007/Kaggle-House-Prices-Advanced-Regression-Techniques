import pandas as pd
import numpy as np
from sklearn import preprocessing

train_data = pd.read_csv('data/train.csv.gz')
test_data = pd.read_csv('data/test.csv.gz')

# print("train data")
# train_data.info()
# print(train_data.head())
# print("\n")
# print("test data")
# test_data.info()

n_features = train_data.shape[1]-1
y_data = train_data.loc[:, 'SalePrice'].values
X_data_numeric = train_data.select_dtypes(exclude='object').drop('SalePrice', 1).values
X_data_ordinal = train_data.select_dtypes(include='object').fillna(value='na').values
X_pred_numeric = test_data.select_dtypes(exclude='object').values
X_pred_ordinal = test_data.select_dtypes(include='object').fillna(value='na').values

LE = preprocessing.LabelEncoder()
X_data_ordinal_transf = []
for label in X_data_ordinal.T:
    LE.fit(label)
    X_data_ordinal_transf.append(LE.transform(label))
X_data_ordinal_transf = np.array(X_data_ordinal_transf)
X_data = np.concatenate((X_data_numeric,
                         X_data_ordinal_transf.T),
                        axis=1)
del X_data_ordinal_transf

X_pred_ordinal_transf = []
for label in X_pred_ordinal.T:
    LE.fit(label)
    X_pred_ordinal_transf.append(LE.transform(label))
X_pred_ordinal_transf = np.array(X_pred_ordinal_transf)
X_pred = np.concatenate((X_pred_numeric,
                         X_pred_ordinal_transf.T),
                        axis=1)
del X_pred_ordinal_transf

print(X_data.shape)
print(X_pred.shape)
