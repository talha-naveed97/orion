# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:36:24 2022

@author: cn5076
"""


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from scipy.signal import savgol_filter
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import random 
import pandas as pd
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from scipy.special import eval_legendre
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GroupKFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

seed = 3

np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = '3'
tf.random.set_seed(seed)


data = scio.loadmat('Ramanmilk_prepped.mat')
data.keys()

X= data['X']
cla = data['CLA'].flatten()
iodine = data['Iodine'].flatten()
groups = data['cvseg'].flatten()




def create_model():
    input_layer = Input((X.shape[1], 1))
    x = Conv1D(32, 30, activation='relu')(input_layer)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(1)(x)
    model = Model(input_layer, output_layer)
    model.compile(loss='mse', optimizer='adam')
    return model


df_X = pd.DataFrame(X)
df_X['Group'] = groups
df_X_groups = df_X.groupby(['Group']).mean()
data_mean = df_X_groups.reset_index().drop(['Group'], axis=1).to_numpy()

df_y = pd.DataFrame(iodine)
df_y['Group'] = groups
df_y_groups = df_y.groupby(['Group']).mean()
targets_mean = df_y_groups.reset_index().drop(['Group'], axis=1).to_numpy()




def Augment_normaldist(x, x_loadings, y_loadings, x_scores,sample_count = 50, number_of_components = 13):
    data_augmented = np.full((sample_count, x.shape[1]), np.nan)
    target_augmented = np.full((sample_count), np.nan)
    T = np.random.normal(0, 1, (sample_count, number_of_components))
    f = np.linalg.norm(x_scores, axis=0)
    T = T* f * (1/np.sqrt(len(x_scores)))
    data_augmented = np.dot(T, x_loadings.T)
    target_augmented = np.dot(T, y_loadings.T)
    return data_augmented,target_augmented



pls_aug = PLSRegression(scale = False, n_components = 20)

pls_aug.fit(data_mean, targets_mean)

data, targets = Augment_normaldist(data_mean,pls_aug.x_loadings_,pls_aug.y_loadings_,pls_aug.x_scores_,500,20)

data = data + np.mean(data_mean, axis = 0)
targets = targets + np.mean(targets_mean)

rdlr = ReduceLROnPlateau(patience=30, factor=0.5, min_lr=1e-6, monitor='loss', verbose=0)
es = EarlyStopping(monitor='loss', patience=60)

model = KerasRegressor(build_fn=create_model, epochs = 1000, batch_size = 8, verbose=0)


pipe = Pipeline([
            ('sc', StandardScaler()),
            ('model', model)
        ])

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(pipe, data, targets, cv = 5, fit_params={'model__callbacks': [rdlr,es]} ,return_times=True,
                                                                      scoring = 'neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

print('Train Sizes:', train_sizes)


print('Train Scores:', train_scores)



print('Train Scores:', test_scores)

fig_name = "plsr_augmented_fig_" + str(seed) + str(1) + ".pdf"

#
# Calculate training and test mean and std
#
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
#
# Plot the learning curve
#
plt.figure()
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy (RMSE)')
plt.grid()
plt.legend(loc='lower right')
plt.show()
plt.savefig(fig_name)

fig_name = "plsr_augmented_fig_" + str(seed) + str(2) + ".pdf"

train_mean = -1 * np.mean(train_scores, axis=1)
train_std = -1 * np.std(train_scores, axis=1)
test_mean = -1 * np.mean(test_scores, axis=1)
test_std = -1 * np.std(test_scores, axis=1)

#
# Plot the learning curve
#
plt.figure()
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy(RMSE)')
plt.grid()
plt.legend(loc='upper right')
plt.show()
plt.savefig(fig_name)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', test_scores)



train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -test_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))


