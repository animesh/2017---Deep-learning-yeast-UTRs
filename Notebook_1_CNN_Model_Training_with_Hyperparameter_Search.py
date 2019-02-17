import pandas as pd
import numpy as np
import scipy
import scipy.stats
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.fmin as hypfmin
import keras
import theano
import random
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

keras.__version__
theano.__version__

base_dir = '/home/animeshs/animeshs/Desktop/2017---Deep-learning-yeast-UTRs/'
data_dir = base_dir + 'Data/'
results_dir = base_dir + 'Results/'
figures_dir = base_dir + 'Figures/'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

if not os.path.exists(figures_dir):
    os.mkdir(figures_dir)

model_name = 'Random_UTR_CNN'
model_params_dir = 'Results/{0}.Hyperparam.Opt/'.format(model_name)

if not os.path.exists(model_params_dir):
    os.mkdir(model_params_dir)

data = pd.read_csv(data_dir + 'Random_UTRs.csv')

def one_hot_encoding(df, seq_column, expression):
    bases = ['A','C','G','T']
    base_dict = dict(zip(bases,range(4))) # {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}
    n = len(df)
    total_width = df[seq_column].str.len().max() + 20
    X = np.zeros((n, 1, 4, total_width))
    seqs = df[seq_column].values
    for i in range(n):
        seq = seqs[i]
        for b in range(len(seq)):
            X[i, 0, base_dict[seq[b]], int(b + round((total_width - len(seq))/2.))] = 1.
        if (i%10000)==0:
            print(i),
    X = X.astype(theano.config.floatX)
    Y = np.asarray(df[expression].values,
                   dtype = theano.config.floatX)[:, np.newaxis]
    return X, Y, total_width

X, Y, total_width = one_hot_encoding(data, 'UTR', 'growth_rate')


sorted_inds = data.sort_values('t0').index.values
train_inds = sorted_inds[:int(0.95*len(sorted_inds))] # 95% of the data as the training set
test_inds = sorted_inds[int(0.95*len(sorted_inds)):] # UTRs with most reads at time point 0 as the test set

seed = 0.5
random.shuffle(train_inds, lambda :seed)

hyperparams = {'conv_width' : [9, 13, 17, 25],
               'conv_filters' : [32, 64, 128, 256],
               'conv_layers' : [2, 3, 4],
               'dense_layers' : [1, 2],
               'conv_dropout' : [None, 0.15],
               'dense_dropout' : [None, 0.1, 0.25, 0.5],
               'dense_units' : [32, 64, 128, 256]}

space = {   'conv_width': hp.choice('conv_width', [9, 13, 17, 25]),
            'conv_filters': hp.choice('conv_filters', [32, 64, 128, 256]),
            'conv_layers': hp.choice('conv_layers', [2, 3, 4]),
            'dense_layers': hp.choice('dense_layers', [1, 2]),
            'conv_dropout': hp.choice('conv_dropout',  [None, 0.15]),
            'dense_dropout': hp.choice('dense_dropout', [None, 0.1, 0.25, 0.5]),
            'dense_units': hp.choice('dense_units', [32, 64, 128, 256]),
        }

def create_model(params):
    model = Sequential()
    model.add(Convolution2D(params['conv_filters'],
                            4,
                            params['conv_width'],
                            border_mode = 'valid',
                            input_shape = (1, 4, total_width),
                            activation = 'relu'))
    if params['conv_dropout']:
        model.add(Dropout(p = params['conv_dropout']))
    for i in range(params['conv_layers'] - 1):
        model.add(Convolution2D(params['conv_filters'],
                                1,
                                params['conv_width'],
                                border_mode = 'same',
                                activation = 'relu'))

        if params['conv_dropout']:
            model.add(Dropout(params['conv_dropout']))

    model.add(Flatten())
    for i in range(params['dense_layers']):
        model.add(Dense(output_dim = params['dense_units'],
                        activation = 'relu'))
        if params['dense_dropout']:
            model.add(Dropout(p = params['dense_dropout']))
    model.add(Dense(output_dim = 1))
    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['mean_squared_error'])
    return model

def f_nn(params):
    model = create_model(params)
    earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  patience = 1,
                                                  verbose = 0,
                                                  mode = 'auto')
    history = keras.callbacks.History()
    global n
    print("\n", n)
    n+=1
    print(params)
    model.fit(X[train_inds],
              Y[train_inds],
              validation_split = 0.2,
              callbacks = [earlyStopping, history],
              verbose = 0,
              nb_epoch = 100)

    print('MSE:',earlyStopping.best)
    return {'loss': earlyStopping.best, 'status': STATUS_OK}

n = 0

#https://github.com/keras-team/keras/issues/3945#issuecomment-281312732
from keras import backend as K
K.set_image_dim_ordering('th')
#for ordering error karas 1->2

trials = Trials()
best = hypfmin(f_nn, space,
               algo = tpe.suggest,
               max_evals = 50,
               trials = trials)
print('best: ')
print(best)

with open(model_params_dir + 'hyperparam_test.pkl', 'w') as f:
    pickle.dump(trials.trials, f)

opt_params = {}

for p in best:
    opt_params[p] = hyperparams[p][best[p]]


opt_params

model = create_model(opt_params)

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience = 0,
                                              verbose = 0,
                                              mode = 'auto')

history = keras.callbacks.History()

modelcheckpoint = keras.callbacks.ModelCheckpoint(model_params_dir + 'model_weights.hdf5',
                                                  monitor = 'val_loss',
                                                  verbose = 0,
                                                  save_best_only = True,
                                                  mode = 'auto')

model.fit(X[train_inds],
          Y[train_inds],
          validation_split = 0.1,
          callbacks = [earlyStopping,
                       history,
                       modelcheckpoint],
          verbose=1,
          nb_epoch = 100)

json_string = model.to_json()
open(model_params_dir + 'model_arch.json', 'w').write(json_string)
model.save_weights(model_params_dir + 'model_weights.h5')


Y_pred = model.predict(X,verbose=1)

print scipy.stats.pearsonr(Y[train_inds].flatten(),
                           Y_pred[train_inds].flatten())[0]**2
print scipy.stats.pearsonr(Y[test_inds].flatten(),
                           Y_pred[test_inds].flatten())[0]**2

model = keras.models.model_from_json(open(model_params_dir + 'model_arch.json').read())
model.load_weights(model_params_dir + 'model_weights.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam')

Y_pred = model.predict(X, verbose=1)

print scipy.stats.pearsonr(Y[train_inds].flatten(),
                           Y_pred[train_inds].flatten())[0]**2

print scipy.stats.pearsonr(Y[test_inds].flatten(),
                           Y_pred[test_inds].flatten())[0]**2


x = Y_pred[test_inds].flatten()
y = Y[test_inds].flatten()

r2 = scipy.stats.pearsonr(x, y)[0]**2

g = sns.jointplot(x,
                  y,
                  stat_func = None,
                  kind = 'scatter',
                  s = 5,
                  alpha = 0.1,
                  size = 5)

g.ax_joint.set_xlabel('Predicted log$_2$ Growth Rate')
g.ax_joint.set_ylabel('Measured log$_2$ Growth Rate')


text = "R$^2$ = {:0.2}".format(r2)
plt.annotate(text, xy=(-5.5, 0.95), xycoords='axes fraction')
plt.title("CNN predictions vs. test set", x = -3, y = 1.25)
