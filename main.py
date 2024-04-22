# main.py
import sys
from data_processing import preprocess_data
from svm import run_svm_with_randomized_search_and_timeout
from neuralnetwork import run_ann_with_timeout
from random_forest import rf_with_timeout
import random
import time

def random_parameters(parameters_grid):
    selected_parameters = {}
    for param, values in parameters_grid.items():
        selected_parameters[param] = random.choice(values)
    return selected_parameters

def parse_command_line_arguments():
    file_path = sys.argv[1] 
    total_time = sys.argv[2]
    single_iter_time = sys.argv[3]

    return file_path, float(total_time), float(single_iter_time)#, base_algorithm, scale_rule, smote_rule, enable_selected_features

def train_and_run(file_path, base_algorithm, X_train, X_test, y_train, y_test, iter_start_time, single_iter_time):
    timeout = single_iter_time - (time.time()-iter_start_time)
    if (base_algorithm=='random_forest'):
        rf_with_timeout(file_path, X_train, X_test, y_train, y_test, timeout)
    elif (base_algorithm=='svm'):
        run_svm_with_randomized_search_and_timeout(X_train, X_test, y_train, y_test, single_iter_time, iter_start_time)
    elif (base_algorithm=='neural_network'):
        run_ann_with_timeout(X_train, X_test, y_train, y_test, timeout)

def total_grid_search(file_path, total_time, single_iter_time):
    start_time = time.time()
    parameters_grid = {
        'base_algo': ['random_forest', 'svm', 'neural_network'],
        'scale_rule': ['StandardScaler', 'RobustScaler', 'PowerTransformer', None],
        'smote_rule' : ['smote', 'smote_enn', None],
        'enable_selected_features': [1, 0]
        # Add more parameters as needed
    }   
    while(1):
        iteration_start_time = time.time()
        if(time.time() - start_time > total_time):
            print('Total Grid Search time limit reached.\n')
            return
        selected_parameters = random_parameters(parameters_grid)
        base_algorithm = selected_parameters['base_algo']
        scale_rule = selected_parameters['scale_rule']
        smote_rule = selected_parameters['smote_rule']
        enable_selected_features = selected_parameters['enable_selected_features']

        rf = ''
        if(enable_selected_features):
            rf = ' random forest feature selection,'
        else:
            rf = 'out random forest feature selection,'

        print('Preprocessing data for base_algorithm= ', base_algorithm, ' with', rf, 'scale_rule=', scale_rule, ' smote_rule=', smote_rule, '\n')

        # NOTE: We hardcoded selected_features so that it's preset and saves run-time.
        X_train, X_test, y_train, y_test = preprocess_data(file_path, scale_rule, smote_rule, enable_selected_features, target='Class')

        train_and_run(file_path, base_algorithm, X_train, X_test, y_train, y_test, iteration_start_time, single_iter_time)

if __name__ == "__main__":
    file_path, total_time, single_iter_time = parse_command_line_arguments()
    print("Total time: ", total_time, "\niteration time: ", single_iter_time, "\n")
    total_grid_search(file_path, total_time, single_iter_time)
    