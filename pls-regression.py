#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split, LeaveOneOut
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_predict
import sklearn



data = scio.loadmat('Ramanmilk_prepped.mat')

X = data['X']
cla = data['CLA'].flatten()
iodine = data['Iodine'].flatten()
groups = data['cvseg'].flatten()

a = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
train, test = next(a.split(X, iodine, groups))
X_train, y_train = X[train], iodine[train]
X_test, y_test = X[test], iodine[test]


print(X_train.shape)
print(X_test.shape)

estimator = PLSRegression()

param_grid = [{'n_components': range(1, 25, 1)}]

gsmse = GridSearchCV(estimator=estimator,
                  param_grid=param_grid,
                  scoring= 'neg_mean_squared_error',
                  cv=10,
                  n_jobs=-1)

estimator = PLSRegression()

param_grid = [{'n_components': range(1, 25, 1)}]

gsr = GridSearchCV(estimator=estimator,
                  param_grid=param_grid,
                  scoring= 'r2',
                  cv=10,
                  n_jobs=-1)

estimator = PLSRegression()

param_grid = [{'n_components': range(1, 25, 1)}]

gsrmse = GridSearchCV(estimator=estimator,
                  param_grid=param_grid,
                  scoring= 'neg_root_mean_squared_error',
                  cv=10,
                  n_jobs=-1)


res_r2 = gsr.fit(X_train, y_train)

res_rmse = gsrmse.fit(X_train, y_train)

res_mse = gsmse.fit(X_train, y_train)

print("R2 Best Params:")
print(gsr.best_score_, gsr.best_params_)

print("RMSE Best Params:")
print(gsrmse.best_score_, gsrmse.best_params_)

print("MSE Best Params:")
print(res_mse.best_score_, res_mse.best_params_)


def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))   

display_cv_results(res_r2)


display_cv_results(res_mse)

display_cv_results(res_rmse)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

data_rmse = []

for i in np.arange(1, 25):
    pls = PLSRegression(n_components=i)
    score = -1*model_selection.cross_val_score(pls, scale(X), y, cv=cv,
               scoring='neg_root_mean_squared_error').mean()
    mse.append(score)


def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores
#     print(y.shape)
#     print(y_cv.flatten().shape)
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv.flatten())
    rpd = y.std()/np.sqrt(mse)
    rmse = np.sqrt(mse)
    return (y_cv, r2, mse, rpd, rmse)

# test with 30 components
r2s = []
mses = []
rpds = []
rmses = []
xticks = np.arange(1, 31)
for n_comp in xticks:
    y_cv, r2, mse, rpd,rmse = optimise_pls_cv(X_train, y_train, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)
    rmses.append(rmse)


# Plot the mses
def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()

plot_metrics(mses, 'MSE', 'min')


plot_metrics(rmses, 'RMSE', 'min')

plot_metrics(rpds, 'RPD', 'max')

plot_metrics(r2s, 'R2', 'max')

y_cv, r2, mse, rpd, rmse = optimise_pls_cv(X_train, y_train, 21)

print('R2: %0.4f, MSE: %0.4f, RPD: %0.4f, RMSE: %0.4f' %(r2, mse, rpd,rmse))


pls = PLSRegression(n_components=21)
pls.fit(X_train, y_train)
yt_predict = pls.predict(X_train)
ytest_predict = pls.predict(X_test)


print(r2_score(y_train, yt_predict))
print(r2_score(y_test, ytest_predict))


mse_train = mean_squared_error(y_train, yt_predict.flatten())
mse_test = mean_squared_error(y_test, ytest_predict.flatten())

print(mse_train)
print(mse_test)

print(np.sqrt(mse_train))
print(np.sqrt(mse_test))




