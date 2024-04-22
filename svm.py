# svm.py
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
import numpy as np
import concurrent.futures
import time

def svm_train_with_timeout(X_train, y_train, X_test, y_test, params, timeout):
    """Train and evaluate an SVM model with a timeout."""
    def train_and_evaluate():
        model = SVC(probability=True, **params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return classification_report(y_test, predictions), accuracy_score(y_test, predictions), auc_score

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(train_and_evaluate)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Training exceeded time limit of {timeout} seconds for parameters: {params}")
            return None, None, None

def run_svm_with_randomized_search_and_timeout(X_train, X_test, y_train, y_test, search_time_limit=600, iter_start_time=0):
    param_distributions = {
    'C': np.logspace(-3, 2, 6),  # Generates values [0.001, 0.01, 0.1, 1, 10, 100]
    'gamma': [0.01, 0.1], # np.logspace(-3, -1, 3),  # Generates values [0.001, 0.01, 0.1]
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']  # Kernel types
}
    param_sampler = ParameterSampler(param_distributions, n_iter=1, random_state=int(time.time()))

    for params in param_sampler:
        print(f"running SVM with parameters: {params}")
        timeout_arg = search_time_limit- (time.time() - iter_start_time)
        report, accuracy, auc = svm_train_with_timeout(X_train, y_train, X_test, y_test, params, timeout=timeout_arg)
        if report is not None:
            print(report)
            print(f"Accuracy: {accuracy}")
            print(f"AUC: {auc}")
        print("--------")