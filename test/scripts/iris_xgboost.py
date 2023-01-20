import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split


def softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), 1)


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
param = {'max_depth': 4, 'eta': 1, 'objective': 'multi:softmax', 'nthread': 4,
         'eval_metric': 'auc', 'num_class': 3}

num_round = 10
bst = xgb.train(param, dtrain, num_round)
y_pred = bst.predict(xgb.DMatrix(X_test))
y_pred_proba = softmax(bst.predict(xgb.DMatrix(X_test), output_margin=True))

np.savetxt('../data/iris_xgboost_true_prediction.txt', y_pred, delimiter='\t')
np.savetxt('../data/iris_xgboost_true_prediction_proba.txt', y_pred_proba, delimiter='\t')
dump_svmlight_file(X_test, y_test, '../data/iris_test.libsvm')
bst.dump_model('../data/iris_xgboost_dump.json', dump_format='json')
