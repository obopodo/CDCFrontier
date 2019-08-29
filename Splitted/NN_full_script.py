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

from sklearn.neural_network import MLPRegressor

import dill
import datetime


# get_ipython().run_line_magic('matplotlib', 'inline')

# dill.load_session('Full_SVR.db')

SEED = 73 # random seed

# Methods defining:

def logging(description='', out=''):
    with open('results/NN/NN_log.txt', 'a') as out_file:
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
    plt.title('NeuralNet predictions on ' + data_type)
    plt.savefig('results/NN/' + current_target + '_' + data_type + '_pred_NN_' + '.png', bbox_inches='tight', dpi=300)
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

    y = data[current_target]
    X = data.drop(current_target, axis=1)

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
    model = MLPRegressor(hidden_layer_sizes=(300,), solver='sgd', validation_fraction=0.25,
                     n_iter_no_change=50, max_iter=5000, early_stopping=True,
                     learning_rate='adaptive', alpha=1e-5, activation='logistic')

    # Rough grid search:
	
	
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = {
        'hidden_layer_sizes': [(9, 4)],
		'alpha': np.geomspace(0.00001, 0.1, 5)
    }

    gs[current_target] = GridS(model, grid)

    gs[current_target].fit(X_st_train, y_train)

    logging('Rough tuning best parameters', str(gs[current_target].best_params_))    

    HL_sizes, alpha = gs[current_target].best_params_['hidden_layer_sizes'], gs[current_target].best_params_['alpha']

    # Fine tuning:

    grid = {
        'hidden_layer_sizes': HL_sizes,
		'alpha': np.geomspace(alpha/5, alpha*5, 10)
    }

    gs[current_target] = GridS(model, grid)

    gs[current_target].fit(X_st_train, y_train)
    
    logging('Fine tuning', str(gs[current_target].best_params_))

    model = gs[current_target].best_estimator_
	
    # Test on train:

    train_pred = model.predict(X_st_train)
    comparsion_plot(y_train, train_pred, data_type='train', current_target=current_target)
    logging('Test on train')
    print_metrics(y_train, train_pred)
    logging()
    # Test on test

    predictions = model.predict(X_st_test)
    comparsion_plot(y_test, predictions, data_type='test', current_target=current_target)
    logging('Test on test')
    print_metrics(y_test, predictions)
    logging()

with open('NN_model.cached', 'wb') as file:
    dill.dump(gs, file)

with open('NN_scalers.cached', 'wb') as file:
    dill.dump(st_scalers, file)

dill.dump_session('Full_NN.db')