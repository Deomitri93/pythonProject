import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


# Loading data from input files
def load_data(drct):
    tr_mcc_codes = pd.read_csv(drct + '\\tr_mcc_codes.csv', sep=';', index_col='mcc_code')
    tr_types = pd.read_csv(drct + '\\tr_types.csv', sep=';', index_col='tr_type')

    trns = pd.read_csv(drct + '\\transactions.csv', index_col='customer_id')
    gndr_train = pd.read_csv(drct + '\\gender_train.csv', index_col='customer_id')
    gndr_test = pd.read_csv(drct + '\\gender_test.csv', index_col='customer_id')
    trns_train = trns.join(gndr_train, how='inner')
    trns_test = trns.join(gndr_test, how='inner')

    del trns

    return trns_train, trns_test, gndr_train, gndr_test


# Cross-validation score (average value ROC AUC metric on train data)
def cv_score(params, train, y_true):
    cv_res = xgb.cv(params, xgb.DMatrix(train, y_true),
                    early_stopping_rounds=10, maximize=True,
                    num_boost_round=10000, nfold=5, stratified=True)
    index_argmax = cv_res['test-auc-mean'].argmax()
    print('Cross-validation, ROC AUC: {:.3f}+-{:.3f}, Trees: {}'.format(cv_res.loc[index_argmax]['test-auc-mean'],
                                                                        cv_res.loc[index_argmax]['test-auc-std'],
                                                                        index_argmax))


# Model building + test classification results returning
def fit_predict(params, num_trees, train, test, target):
    params['learning_rate'] = params['eta']
    clf = xgb.train(params, xgb.DMatrix(train.values, target, feature_names=list(train.columns)),
                    num_boost_round=num_trees, maximize=True)
    y_pred = clf.predict(xgb.DMatrix(test.values, feature_names=list(train.columns)))
    submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])
    return clf, submission


# Variable importance drawing. Variable importance - number of sample splits,
# where the variable exists. The bigger number - the better variable is
def draw_feature_importance(clf, top_k=10):
    plt.figure(figsize=(10, 10))

    importance = dict(sorted(clf.get_score().items(), key=lambda x: x[1])[-top_k:])
    y_pos = np.arange(len(importance))

    plt.barh(y_pos, list(importance.values()), align='center', color='green')
    plt.yticks(y_pos, importance.keys(), fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel('Feature importance', fontsize=15)
    plt.title('Features importance, Sberbank Gender Prediction', fontsize=18)
    plt.ylim(-0.5, len(importance) - 0.5)
    plt.show()


# Return model params
def model_params():
    prms = {
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,

        'gamma': 0,
        'lambda': 0,
        'alpha': 0,
        'min_child_weight': 0,

        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'njobs': -1,
        'tree_method': 'approx'
    }
    return prms


def features_creation_basic(x):
    features = []
    features.append(pd.Series(x[x['amount'] > 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount'] < 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('negative_transactions_')))
    features.append(pd.Series(x[x['amount'] > 0]['mcc_code'].agg(['median'])
                              .add_prefix('positive_transactions_mcc_code')))
    features.append(pd.Series(x[x['amount'] < 0]['mcc_code'].agg(['median'])
                              .add_prefix('negative_transactions_mcc_code_')))
    features.append(pd.Series(x[x['amount'] > 0]['tr_type'].agg(['median'])
                              .add_prefix('positive_transactions_tr_type_')))
    features.append(pd.Series(x[x['amount'] < 0]['tr_type'].agg(['median'])
                              .add_prefix('negative_transactions_tr_type_')))

    return pd.concat(features)


def features_creation_advanced(x):
    features = []
    features.append(pd.Series(x['day'].value_counts(normalize=True).add_prefix('day_')))
    features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['night'].value_counts(normalize=True).add_prefix('night_')))

    features.append(pd.Series(x[x['amount'] > 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount'] < 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('negative_transactions_')))
    features.append(pd.Series(x[x['amount'] > 0]['mcc_code'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('positive_transactions_mcc_code_')))
    features.append(pd.Series(x[x['amount'] < 0]['mcc_code'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('negative_transactions_mcc_code_')))
    features.append(pd.Series(x[x['amount'] > 0]['tr_type'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('positive_transactions_tr_type_')))
    features.append(pd.Series(x[x['amount'] < 0]['tr_type'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
                              .add_prefix('negative_transactions_tr_type_')))
    print(features)

    return pd.concat(features)