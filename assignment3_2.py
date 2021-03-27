#------------------------------------
#import libraries
#------------------------------------
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest

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
x_train = train.iloc[:,:300].values
y_train = train.iloc[:,ncol_train-1].values
x_test = test.iloc[:,:300].values
y_test = test.iloc[:,ncol_test-1].values

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
#creation of nan
#------------------------------------

#array with the index of attributes in which we will include nan
nan_col_paramsize = np.floor(x_train.shape[1]*0.1) #10% of the columns
nan_col_paramsize = int(nan_col_paramsize)
np.random.seed(nia) #reproducibility
nan_col = np.random.choice(x_train.shape[1], size=nan_col_paramsize, replace=False)

#creation of nan
for col in nan_col:

    #creating missing values in the train data set
    nan_row_paramsize = np.floor(x_train.shape[0]*0.1) #10% of the rows
    nan_row_paramsize = int(nan_row_paramsize)
    np.random.seed(nia) #reproducibility
    nan_row = np.random.choice(x_train.shape[0], size=nan_row_paramsize, replace=False)
    for row in nan_row:
        x_train[row, col] = np.nan

    #creating missing values in the test data set
    nan_row_paramsize = np.floor(x_test.shape[0]*0.1) #10% of the rows
    nan_row_paramsize = int(nan_row_paramsize)
    np.random.seed(nia) #reproducibility
    nan_row = np.random.choice(x_test.shape[0], size=nan_row_paramsize, replace=False)
    for row in nan_row:
        x_test[row, col] = np.nan

#------------------------------------
#pipeline 1: PCA
#------------------------------------
imputer = preprocessing.Imputer(strategy='median', axis=0)
feature_sel = VarianceThreshold(threshold=0.0)
scaler = preprocessing.MinMaxScaler()
pca = PCA()
knn = neighbors.KNeighborsRegressor()

#pipeline definition
pipeline_pca = Pipeline([
    ('impute', imputer),
    ('feature_selection', feature_sel),
    ('scale', scaler),
    ('pca', pca),
    ('knn_reg', knn)
])

#hyper parameter tuning pca
paramgrid = {
    'pca__n_components': list(range(1,40))
}

#search for the best number of components
clf_pca = GridSearchCV(
    pipeline_pca,
    param_grid=paramgrid,
    scoring='neg_mean_squared_error',
    cv=x_train_val_partition,
    verbose=0
)

clf_pca.fit(x_train, y_train)

#setting the best value for the number of components
pipeline_pca = pipeline_pca.set_params(**{'pca__n_components': clf_pca.best_params_['pca__n_components']})

print('-------------------------')
print('K NEIGHBORS PCA')
print('-------------------------')
print('---hyper parameter tunning---')
print('parameters: pca__n_components={0}'.format(clf_pca.best_params_['pca__n_components']))

#hyper parameter tuning knn
paramgrid = {
    'knn_reg__n_neighbors': list(range(2,40))
}

#search for the best number of components
clf_pca = GridSearchCV(
    pipeline_pca,
    param_grid=paramgrid,
    scoring='neg_mean_squared_error',
    cv=x_train_val_partition,
    verbose=0
)

#final model 
clf_pca.fit(x_train, y_train)
y_test_pred_pca = clf_pca.predict(x_test)
clf_pca_r2 = metrics.r2_score(y_test, y_test_pred_pca)

print('parameters: knn_reg__n_neighbors={0}'.format(clf_pca.best_params_['knn_reg__n_neighbors']))
print('coefficient of determination: {0:.3f}'.format(clf_pca_r2))

#------------------------------------
#pipeline 2: selectKBest
#------------------------------------
imputer = preprocessing.Imputer(strategy='median', axis=0)
feature_sel = VarianceThreshold(threshold=0.0)
scaler = preprocessing.MinMaxScaler()
kbest = SelectKBest()
knn = neighbors.KNeighborsRegressor()

#pipeline definition
pipeline_kbest = Pipeline([
    ('impute', imputer),
    ('feature_selection', feature_sel),
    ('scale', scaler),
    ('kbest', kbest),
    ('knn_reg', knn)
])

#hyper parameter tuning k best
paramgrid = {
    'kbest__k': list(range(1,40))
}

#search for the best number of components
clf_kbest = GridSearchCV(
    pipeline_kbest,
    param_grid=paramgrid,
    scoring='neg_mean_squared_error',
    cv=x_train_val_partition,
    verbose=0
)

clf_kbest.fit(x_train, y_train)

print('-------------------------')
print('K NEIGHBORS KBest')
print('-------------------------')
print('---hyper parameter tunning---')
print('parameters: kbest__k={0}'.format(clf_kbest.best_params_['kbest__k']))

#setting the best value for the number of components
pipeline_kbest = pipeline_kbest.set_params(**{'kbest__k': clf_kbest.best_params_['kbest__k']})

#hyper parameter tuning knn
paramgrid = {
    'knn_reg__n_neighbors': list(range(2,100))
}

#search for the best number of components
clf_kbest = GridSearchCV(
    pipeline_kbest,
    param_grid=paramgrid,
    scoring='neg_mean_squared_error',
    cv=x_train_val_partition,
    verbose=0
)

#final model 
clf_kbest.fit(x_train, y_train)
y_test_pred_kbest = clf_kbest.predict(x_test)
clf_kbest_r2 = metrics.r2_score(y_test, y_test_pred_kbest)

print('parameters: knn_reg__n_neighbors={0}'.format(clf_kbest.best_params_['knn_reg__n_neighbors']))
print('coefficient of determination: {0:.3f}'.format(clf_kbest_r2))
