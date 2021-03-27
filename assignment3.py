#------------------------------------
#import libraries
#------------------------------------
from numpy.random import randint
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree, neighbors, svm
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn import metrics
from scipy.stats import uniform

#------------------------------------
#import data set
#------------------------------------

train_filepath = ... 
test_filepath = ...
train = pd.read_csv(train_filepath, sep=',')
test = pd.read_csv(test_filepath, sep=',')

#------------------------------------
#summary of data train
#------------------------------------
# train.describe()
# train.info() #variable type: float64: 1194, int64 = 7

# print(train.dtype)
# print(test.dtype)

train_shape = train.shape
test_shape = test.shape

nrow_train = train_shape[0]
ncol_train = train_shape[1]
nrow_test = test_shape[0]
ncol_test = test_shape[1]
#------------------------------------
#transformation of data
#------------------------------------

#transform pandas dataframe into numpy array
x_train = train.iloc[:,:ncol_train-1].values
y_train = train.iloc[:,ncol_train-1].values
x_test = test.iloc[:,:ncol_test-1].values
y_test = test.iloc[:,ncol_test-1].values

#normalization of arrays
scaler = preprocessing.MinMaxScaler() 
scaler.fit_transform(x_train)
scaler.transform(x_test)

#retain first 75 columns
x_train = x_train[:,:75]
x_test = x_test[:,:75]

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#------------------------------------
#parameters to be used
#------------------------------------

#student number indentifier
nia = ...

#validation indices
validation_indices = np.zeros(nrow_train) #index test sample: 0
validation_indices[:10*365] = -1 #index training sample: -1
x_train_val_partition = PredefinedSplit(validation_indices)

#------------------------------------
#regression trees
#------------------------------------

#deafult parameters
clf_tree1 = tree.DecisionTreeRegressor()
np.random.seed(nia) #reproducibility
clf_tree1.fit(x_train, y_train)
y_test_pred_tree1 = clf_tree1.predict(x_test)
clf_tree1_r2 = metrics.r2_score(y_test, y_test_pred_tree1)

#hyper parameter tunning
clf_tree2 = tree.DecisionTreeRegressor()
clf_tree2_paramgrid = {
    'max_depth': list(range(2,20)), 
    'min_samples_split': list(range(2,30))
}
clf_tree2_grid = RandomizedSearchCV(
    clf_tree2,
    param_distributions=clf_tree2_paramgrid,
    n_iter=250,
    scoring='r2',
    cv=x_train_val_partition,
    verbose=0
)
np.random.seed(nia) #reproducibility
clf_tree2_grid.fit(x_train, y_train)
y_test_pred_tree2 = clf_tree2_grid.predict(x_test)
clf_tree2_r2 = metrics.r2_score(y_test, y_test_pred_tree2)

#------------------------------------
#k neighbors
#------------------------------------

#deafult parameters
clf_knn1 = neighbors.KNeighborsRegressor()
np.random.seed(nia) #reproducibility
clf_knn1.fit(x_train, y_train)
y_test_pred_knn1 = clf_knn1.predict(x_test)
clf_knn1_r2 = metrics.r2_score(y_test, y_test_pred_knn1)

#hyper parameter tunning
clf_knn2 = neighbors.KNeighborsRegressor()
clf_knn2_paramgrid = {
    'n_neighbors': list(range(2,40))
}
clf_knn2_grid = RandomizedSearchCV(
    clf_knn2,
    param_distributions=clf_knn2_paramgrid,
    n_iter=20,
    scoring='r2',
    cv=x_train_val_partition,
    verbose=0
)
np.random.seed(nia) #reproducibility
clf_knn2_grid.fit(x_train, y_train)
y_test_pred_knn2 = clf_knn2_grid.predict(x_test)
clf_knn2_r2 = metrics.r2_score(y_test, y_test_pred_knn2)

#------------------------------------
#svr
#------------------------------------

from scipy import stats

#default parameters
clf_svr1 = svm.SVR()
np.random.seed(nia) #reproducibility
clf_svr1.fit(x_train, y_train)
y_test_pred_svr1 = clf_svr1.predict(x_test)
clf_svr1_r2 = metrics.r2_score(y_test, y_test_pred_svr1)

#hyper parameter tunning
clf_svr2 = svm.SVR()
clf_svr2_paramgrid = {"C": stats.uniform(0.5, 10),
             "gamma": stats.uniform(0.001, 1)}

clf_svr2_grid = RandomizedSearchCV(
    clf_svr2,
    param_distributions=clf_svr2_paramgrid,
    n_iter=100,
    scoring='r2',
    cv=x_train_val_partition,
    verbose=0
)
np.random.seed(nia) #reproducibility
clf_svr2_grid.fit(x_train, y_train)
y_test_pred_svr2 = clf_svr2_grid.predict(x_test)
clf_svr2_r2 = metrics.r2_score(y_test, y_test_pred_svr2)

#------------------------------------
#output
#------------------------------------
print('-------------------------')
print('REGRESSION TREE')
print('-------------------------')
print('---default paramters---')
print('Coefficient of determination: {0:.3f}'.format(clf_tree1_r2))
print('---hyper parameter tunning---')
print('parameters: max_depth={0}, min_samples_split={1}'.format(clf_tree2_grid.best_params_['max_depth'], clf_tree2_grid.best_params_['min_samples_split']))
print('coefficient of determination: {0:.3f}'.format(clf_tree2_r2))

print('-------------------------')
print('K NEIGHBORS')
print('-------------------------')
print('---default paramters---')
print('Coefficient of determination: {0:.3f}'.format(clf_knn1_r2))
print('---hyper parameter tunning---')
print('parameters: n_neighbors={0}'.format(clf_knn2_grid.best_params_['n_neighbors']))
print('coefficient of determination: {0:.3f}'.format(clf_knn2_r2))

print('-------------------------')
print('SVR')
print('-------------------------')
print('---default paramters---')
print('Coefficient of determination: {0:.3f}'.format(clf_svr1_r2))
print('---hyper parameter tunning---')
print('parameters: C={0:.3f}, epsilon={1:.3f}'.format(clf_svr2_grid.best_params_['C'], clf_svr2_grid.best_params_['epsilon']))
print('coefficient of determination: {0:.3f}'.format(clf_svr2_r2))
