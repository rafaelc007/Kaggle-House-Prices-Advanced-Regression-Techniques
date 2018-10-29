import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as GBR
from sklearn.model_selection import GridSearchCV


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
    return [X_data, y_data, X_pred]


X_data, y_data, X_pred = get_data()

# Model parameters
lr = [0.8, 0.9, 0.1, 0.15, 0.16]  # learning_rate
n_est = [200, 250, 300]  # number of boosting stages
max_dep = [2, 3, 5]  # max_depth

parameters = {'learning_rate': lr, 'n_estimators': n_est,
              'max_depth': max_dep}


# log scaling the target
y_data = np.log(y_data)

# training a Gradient boosting regressor
model = GBR(criterion='mse')
# model.fit(X_data, y_data)

clf = GridSearchCV(model, parameters, cv=5,
                   scoring='neg_mean_squared_error',
                   n_jobs=4, pre_dispatch=3, verbose=1)
clf.fit(X_data, y_data)
result = pd.DataFrame(clf.cv_results_)
result.to_csv('data/result.csv', index=False)
print('best params')
print(result.loc[result['mean_test_score'].idxmax(), 'params'])
print('mean test: ', result['mean_test_score'].max())
print('std: ', result.loc[result['mean_test_score'].idxmax(),
                          'std_test_score'])

# evaluate model
# scored = cross_val_score(model, X_data, y=y_data, cv=5)
# prices = np.round(np.exp(model.predict(X_pred)), 2)
# Gen_output_file(test_Ids, prices)
# print('scored {0}'.format(np.mean(scored)))
