# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 00:13:19 2022

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

seed = 42

np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = '42'
tf.random.set_seed(seed)

session_conf = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )
sess = tf.compat.v1.Session( graph=tf.compat.v1.get_default_graph(), config=session_conf )
tf.compat.v1.keras.backend.set_session(sess)

data = scio.loadmat('Ramanmilk_prepped.mat')
data.keys()

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
data = df_X_groups.reset_index().drop(['Group'], axis=1).to_numpy()

df_y = pd.DataFrame(iodine)
df_y['Group'] = groups
df_y_groups = df_y.groupby(['Group']).mean()
targets = df_y_groups.reset_index().drop(['Group'], axis=1).to_numpy()

data.shape

targets.shape

rdlr = ReduceLROnPlateau(patience=30, factor=0.5, min_lr=1e-6, monitor='loss', verbose=1)
es = EarlyStopping(monitor='loss', patience=60)

model = KerasRegressor(build_fn=create_model, epochs = 500, batch_size = 8, verbose=0)


pipe = Pipeline([
            ('sc', StandardScaler()),
            ('model', model)
        ])

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(pipe, data, targets, cv = 5 , return_times=True,
                                                                      scoring = 'neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

print('Train Sizes:', train_sizes)



print('Train Scores:', train_scores)



print('Train Scores:', test_scores)

fig_name = "fig_" + str(seed) + str(1) + ".pdf"

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
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()
plt.savefig(fig_name)

fig_name = "fig_" + str(seed) + str(2) + ".pdf"

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
plt.ylabel('Model accuracy')
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


