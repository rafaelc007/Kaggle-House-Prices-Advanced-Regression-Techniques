import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as GBR


def Gen_output_file(Id, Price):
    Id = Id.astype(int)
    # Price = Price.reshape(-1, 1)
    ans_data = pd.DataFrame({'Id': pd.Series(Id, dtype=int),
                             'SalePrice': pd.Series(Price,
                                                    dtype=float)})
    ans_data.to_csv('./data/ans_GBR.csv', index=False)
    print('Answer saved in disc!')


def preprocess_X(X):
    LE = preprocessing.LabelEncoder()
    X_ordinal_transf = []
    for label in X.T:
        LE.fit(label.astype(str))
        X_ordinal_transf.append(LE.transform(label))
    return np.array(X_ordinal_transf)


def get_data():
    # import data
    train_data = pd.read_csv('data/train.csv.gz')
    test_data = pd.read_csv('data/test.csv.gz')

    # split features from target
    test_Ids = test_data.loc[:, 'Id'].values
    train_data = train_data.drop('Id', 1)
    test_data = test_data.drop('Id', 1)

    y_data = train_data.loc[:, 'SalePrice'].values
    X_data_numeric = train_data.select_dtypes(exclude='object').drop('SalePrice', 1).fillna(value=999).values
    X_data_ordinal = train_data.select_dtypes(include='object').fillna(value='na').values
    X_pred_numeric = test_data.select_dtypes(exclude='object').fillna(value=999).values
    X_pred_ordinal = test_data.select_dtypes(include='object').fillna(value='na').values

    # preprocess x_train
    X_data_ordinal_transf = preprocess_X(X_data_ordinal)
    X_data = np.concatenate((X_data_numeric,
                             X_data_ordinal_transf.T),
                            axis=1)

    # preprocess x_pred
    X_pred_ordinal_transf = preprocess_X(X_pred_ordinal)
    X_pred = np.concatenate((X_pred_numeric,
                             X_pred_ordinal_transf.T),
                            axis=1)
    return [X_data, y_data, X_pred, test_Ids]


X_data, y_data, X_pred, test_Ids = get_data()


# Model parameters
lr = 0.15  # learning rate
n_est = 200  # number of boosting stages


# log scaling the target
y_data = np.log(y_data)

# training a Gradient boosting regressor
model = GBR(learning_rate=lr, n_estimators=n_est,
            random_state=0)
model.fit(X_data, y_data)

# evaluate model

scored = cross_val_score(model, X_data, y=y_data,
                         cv=5, scoring='neg_mean_squared_error',
                         n_jobs=2)
prices = np.round(np.exp(model.predict(X_pred)), 2)
Gen_output_file(test_Ids, prices)
print('scored {0}'.format(np.mean(scored)))
