import pandas as pd
import numpy as np
# import os

# import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# from numpy.random import normal, uniform
# import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import dill
import datetime


# get_ipython().run_line_magic('matplotlib', 'inline')

# dill.load_session('Full_SVR.db')

SEED = 73 # random seed

# Methods defining:

def logging(description='', out=''):
    with open('results/SVR/SVR_log.txt', 'a') as out_file:
        out_file.write(str(description) + ': ' + str(out) + '\n')
    
    print(description + ': ' + str(out))

# def clear_log():
    # with open('results/NN/NN_log.txt', 'w') as out_file:
        # out_file.write('log file have been cleared ' + str(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')))

# clear_log()

def comparsion_plot(true_values, predictions, current_target, data_type = 'test'):
    my_plot = plt.scatter(true_values, predictions)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r')
    plt.xlabel(current_target + 'True Values')
    plt.ylabel(current_target + 'Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.title('SVR predictions on ' + data_type)
    plt.savefig('results/SVR/' + current_target + '_' + data_type + '_pred_SVR_' + '.png', bbox_inches='tight', dpi=300)
    plt.clf()
    # plt.show()

def print_metrics(true_val, predicted_val):
    logging('r', np.corrcoef(true_val, predicted_val)[0,1].round(3))
    logging('MSE', mean_squared_error(true_val, predicted_val).round(3))
    logging('R2', r2_score(true_val, predicted_val).round(3))


def GridS(model, grid, **kwargs):
    return GridSearchCV(model, grid, 
                      n_jobs=-1, 
                      scoring=['neg_mean_squared_error', 'r2'], 
                      refit='neg_mean_squared_error', 
                      cv=cv, 
                      verbose=1)
###################################################################################

# Data reading:
data = pd.read_csv('data/prepared/data_processed.csv')
names = data.columns

gs = {} # dict of GS-tuned models
st_scalers = {} # dict of scalers for further usage .transform() method

# Current time log:
logging('Running datetime', str(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')))

'''
MAIN BODY:
cycle by gene names
'''

targets = []

for name in names:
	if not name.startswith('FT'):
		targets.append(name)
		
targets.sort()

for name in targets:
    current_target = name
 
    logging('target', current_target)

    y = data[current_target].values
    X = data.drop(current_target, axis=1).values

    # Data standartization:
    st_scalers[current_target] = StandardScaler()
    X_st = st_scalers[current_target].fit_transform(X)

    # Train-test split: 
    X_st_train, X_st_test, y_train, y_test = train_test_split(X, y, \
        test_size = 0.25, random_state=SEED, shuffle=True)

    '''
    SVR model
    Model building:
    '''
    svm_reg = SVR(kernel='rbf', epsilon=1)

    # Rough grid search:

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = {
        'gamma': np.geomspace(0.0001, 10, 6),\
        'C': np.geomspace(0.0001, 1000, 8)
        #'epsilon': np.geomspace(0.001, 10, 5)
        #'epsilon': 1.0
    }

    gs[current_target] = GridS(svm_reg, grid)

    gs[current_target].fit(X_st_train, y_train)

    logging('Rough tuning best params', str(gs[current_target].best_params_))    

    gamma, C = gs[current_target].best_params_['gamma'],\
        gs[current_target].best_params_['C']

    # Fine tuning:

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = {
        'gamma': np.geomspace(gamma/5, gamma*5, 10),
        'C': np.geomspace(C/5, C*5, 10)
        #'epsilon': np.geomspace(epsilon/5, epsilon*5, 10)
        #'epsilon': 1
    }

    gs[current_target] = GridS(svm_reg, grid)
	
    gs[current_target].fit(X_st_train, y_train)
    
    logging('Fine tuning best params', str(gs[current_target].best_params_))

    svr = gs[current_target].best_estimator_

    logging('Number of SVs', len(svr.support_))
    logging('Number of training vectors', len(X_st_train))

    # Test on train:

    train_pred = svr.predict(X_st_train)

    comparsion_plot(y_train, train_pred, data_type='train', current_target=current_target)
    logging('Test on train')
    print_metrics(y_train, train_pred)
    logging()
    # Test on test

    predictions = svr.predict(X_st_test)
    comparsion_plot(y_test, predictions, data_type='test', current_target=current_target)
    logging('Test on test')
    print_metrics(y_test, predictions)

    logging()

with open('SVR_model.cached', 'wb') as file:
    dill.dump(gs, file)

with open('SVR_scalers.cached', 'wb') as file:
    dill.dump(st_scalers, file)

dill.dump_session('Full_SVR.db')