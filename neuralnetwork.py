# neuralnetwork.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score
from tensorflow.keras.initializers import HeNormal
import random
import concurrent.futures

def random_parameters(parameters_grid):
    selected_parameters = {}
    for param, values in parameters_grid.items():
        selected_parameters[param] = random.choice(values)
    return selected_parameters

def run_ann_with_timeout(X_train, X_test, y_train, y_test, time_limit):

    def run_ann():
        parameters_grid = {
        'kernel': ['HeNormal', 'uniform', 'normal'],
        'optimizer': ['adam', 'SGD', 'adagrad', 'adamax', 'nadam'],
        'batch_size' : [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        'epochs' : [5, 10, 15, 20],
        'loss' : ['binary_crossentropy']
        # Add more parameters as needed
        }
        selected_parameters = random_parameters(parameters_grid)

        kern_init = selected_parameters['kernel']
        sel_optimizer = selected_parameters['optimizer']
        sel_batchsize = selected_parameters['batch_size']
        sel_epochs = selected_parameters['epochs']
        sel_loss = selected_parameters['loss']

        # Defining the model architecture
        if (kern_init == 'HeNormal'):
            classifier = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(units=64, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=32, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=32, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=16, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=8, kernel_initializer=HeNormal(), activation='relu'),
            Dense(units=1, kernel_initializer=HeNormal(), activation='sigmoid')
        ])
        else: #use kern_init as value 'uniform' or 'normal'
            classifier = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(units=64, kernel_initializer=kern_init, activation='relu'),
            Dense(units=32, kernel_initializer=kern_init, activation='relu'),
            Dense(units=32, kernel_initializer=kern_init, activation='relu'),
            Dense(units=16, kernel_initializer=kern_init, activation='relu'),
            Dense(units=8, kernel_initializer=kern_init, activation='relu'),
            Dense(units=1, kernel_initializer=kern_init, activation='sigmoid')
        ])

        print("Running neural network with parameters: ", selected_parameters, "\n...\n")

        classifier.compile(optimizer=sel_optimizer, loss=sel_loss, metrics=['accuracy'])

        # Training the model
        classifier.fit(X_train, y_train, batch_size=sel_batchsize, epochs=sel_epochs, verbose=1)

        # Evaluating the model
        scores = classifier.evaluate(X_test, y_test)
        print("\nModel Accuracy: %.2f%%" % (scores[1]*100))

        # Predicting the test set results
        y_pred = classifier.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)  # Converting probabilities to class labels

        # Generating confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred_classes)
        print("Confusion Matrix:")
        print(cm)

        print("Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=['Non-Fraudulent', 'Fraudulent']))

        # Calculating recall for fraudulent transactions
        print(f"Recall for Fraudulent Transactions: {recall_score(y_test, y_pred_classes):.2f}")

        # Calculating and printing the AUC
        auc = roc_auc_score(y_test, y_pred.ravel())  # Here y_pred is used directly to calculate AUC
        print(f"AUC: {auc:.2f}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_ann)
        try:
            return future.result(timeout=time_limit)
        except concurrent.futures.TimeoutError:
            print(f"Training exceeded time limit of {time_limit} seconds")
            return None, None, None