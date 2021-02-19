import re

from tqdm.notebook import tqdm_notebook
from warnings import filterwarnings

import functions


# Directory, where files are stored
dataDir = 'C:\\Users\\Dmitry\\Downloads\\Python для анализа данных\\Python And Data Analysis\\data'


if __name__ == '__main__':
    # %matplotlib inline
    filterwarnings('ignore')

    transactions_train, transactions_test, gender_train, gender_test = functions.load_data(dataDir)

    params = functions.model_params()

    tqdm_notebook.pandas(desc="Progress:")

    data_train = transactions_train.groupby(transactions_train.index).progress_apply(functions.features_creation_basic)
    data_test = transactions_test.groupby(transactions_test.index).progress_apply(functions.features_creation_basic)

    target = data_train.join(gender_train, how='inner')['gender']
    functions.cv_score(params, data_train, target)

    # Number of trees for XGBoost is important to define after results on cross-validation
    clf, submission = functions.fit_predict(params, 70, data_train, data_test, target)

    functions.draw_feature_importance(clf, 10)

    submission.to_csv('Python And Data Analysis\\data\\basic_features_submission.csv')

    for df in [transactions_train, transactions_test]:
        df['day'] = df['tr_datetime'].str.split().apply(lambda x: int(x[0]) % 7)
        df['hour'] = df['tr_datetime'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
        df['night'] = ~df['hour'].between(6, 22).astype(int)

    data_train = transactions_train.groupby(transactions_train.index).progress_apply(functions.features_creation_advanced)\
        .unstack(-1)
    data_test = transactions_test.groupby(transactions_test.index).progress_apply(functions.features_creation_advanced)\
        .unstack(-1)

    print(data_train)

    target = data_train.join(gender_train, how='inner')['gender']
    functions.cv_score(params, data_train, target)

    # Number of trees for XGBoost is important to define after results on cross-validation
    clf, submission = functions.fit_predict(params, 100, data_train, data_test, target)

    functions.draw_feature_importance(clf, 100)

    submission.to_csv(dataDir + '\\submission_advanced.csv')
