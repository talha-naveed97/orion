# coding: utf-8



import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
import random 
import pandas as pd
from sklearn.metrics import r2_score
import tensorflow as tf
import sklearn
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GroupKFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os




print('The scikit-learn version is {}.'.format(sklearn.__version__))


np.random.seed(3)
random.seed(3)
os.environ['PYTHONHASHSEED'] = '3'
tf.random.set_seed(3)

session_conf = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )
sess = tf.compat.v1.Session( graph=tf.compat.v1.get_default_graph(), config=session_conf )
tf.compat.v1.keras.backend.set_session(sess)



data = scio.loadmat('Ramanmilk_prepped.mat')
print(data.keys())



X= data['X']
cla = data['CLA'].flatten()
iodine = data['Iodine'].flatten()
groups = data['cvseg'].flatten()




def create_model():
    input_layer = Input((X.shape[1], 1))
    x = Conv1D(32, 30, activation='elu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(16, 30, activation='elu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(8, 30, activation='elu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(input_layer, output_layer)
    model.compile(loss='mse', optimizer='adam')
    return model


# Mean by groups

df_X = pd.DataFrame(X)
df_y = pd.DataFrame(iodine)
df_X['Group'] = groups
df_y['Group'] = groups
df_X_groups = df_X.groupby(['Group']).mean()
df_y_groups = df_y.groupby(['Group']).mean()

data = df_X_groups.reset_index().drop(['Group'], axis=1).to_numpy()
targets = df_y_groups.reset_index().drop(['Group'], axis=1).to_numpy()



rdlr = ReduceLROnPlateau(patience=30, factor=0.5, min_lr=1e-6, monitor='loss', verbose=1)


model = KerasRegressor(build_fn=create_model, epochs = 1000, batch_size = 8, verbose=0)
cv = GroupKFold(n_splits=5)



grp_lst = np.unique(groups)




train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, data, targets, cv= 5, fit_params={'callbacks': [rdlr]},return_times=True, scoring = 'neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))


print('Train Sizes:', train_sizes)



print('Train Scores:', train_scores)



print('Train Scores:', test_scores)



plt.figure()
plt.plot(train_sizes,np.mean(train_scores,axis=1))
plt.show()
plt.savefig('fig1.pdf')



plt.figure()
plt.plot(train_sizes,np.mean(test_scores,axis=1))
plt.show()
plt.savefig('fig2.pdf')


plt.figure()
plt.plot(train_sizes,np.mean(train_scores,axis=1), label = 'Train')
plt.plot(train_sizes,np.mean(test_scores,axis=1), label = 'Test')
plt.legend()
plt.show()
plt.savefig('fig3.pdf')




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
plt.savefig('fig4.pdf')


print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', test_scores)




train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -test_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
