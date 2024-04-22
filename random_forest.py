# randomforest.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from selected_features import SELECTED_FEATURES
import joblib
import random
import concurrent.futures

def rf_with_timeout(csv_file_path, X_train, X_test, y_train, y_test, timeout):
    def random_forest_classification():
        print("Running random forest for classification...\n")

        randforest_model =  RandomForestClassifier(random_state=42)

        randforest_model.fit(X_train, y_train)
        y_pred = randforest_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
    
        if report is not None:
            print(report)
            print(f"Accuracy: {accuracy}")
            print(f"AUC: {auc_score}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(random_forest_classification)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Training exceeded time limit of {timeout} seconds")
            return None, None, None

def random_forest_preprocessing_main(self, csv_file_path, num_features_to_select=30, threshold=0.95):
    randforest_model = RandForest(csv_file_path)
    threshold = random.choice([0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995])
    # Preprocess data
    randforest_model.preprocess_data()
    
    # Train the model and print out classification report
    auc_score = randforest_model.train_model()
    print(f"Initial AUC Score: {auc_score}")

    # Evaluate the importance of features based on cross-validated performance
    # trimmed_data, feature_names, num_features = randforest_model.evaluate_feature_importance(n_features_to_select=num_features_to_select, threshold=threshold)  
    
    # print(f"{num_features} important features were found: {feature_names}")

    trimmed_data = self.data[SELECTED_FEATURES]

    # Save the model
   #fraud_detector.save_model('rf_model.pkl')

    return trimmed_data

class RandForest:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.model = RandomForestClassifier(random_state=42)
        self.X = None
        self.y = None
        self.feature_importances = None

    def preprocess_data(self):
        # Assume last column is the label and all others are features
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.feature_importances = self.model.feature_importances_
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return roc_auc_score(y_test, y_pred)
    
    def train_preprocessed(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return roc_auc_score(y_test, y_pred)

    def evaluate_feature_importance(self, n_features_to_select, threshold):
        feature_indices = np.argsort(self.feature_importances)[::-1]
        optimal_num_features = n_features_to_select
        for i in range(1, n_features_to_select + 1):
            selected_features = feature_indices[:i]
            X_reduced = self.X[:, selected_features]
            scores = cross_val_score(self.model, X_reduced, self.y, cv=5, scoring='roc_auc')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"Top {i} features: AUC = {mean_score:.4f} (+/- {std_score:.4f})")
            if (mean_score > threshold):
                optimal_num_features = i
                break
        final_selected_features = feature_indices[:optimal_num_features]
        class_index = self.data.columns.get_loc('Class')
        final_selected_features_with_class = np.append(final_selected_features, class_index)

        important_feature_names = self.data.columns[final_selected_features]
        # if 'Class' not in final_selected_features:
        # final_selected_features.append('Class')
        # trimmed_data_final_features = feature_indices[:optimal_num_features + 'Class']
        # trimmed_data = self.data.iloc[:, trimmed_data_final_features]

        trimmed_data = self.data.iloc[:, final_selected_features_with_class]

        return trimmed_data, important_feature_names, optimal_num_features
            

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)

    def load_model(self, model_file):
        self.model = joblib.load(model_file)

    def make_prediction(self, sample_data):
        scaler = StandardScaler()
        sample_data = scaler.fit_transform(sample_data)
        return self.model.predict(sample_data)

if __name__ == "__main__":

    pass
    # To predict on new sample data:
    # sample_data = pd.DataFrame(...)  # Your new sample data as a DataFrame
    # predictions = fraud_detector.make_prediction(sample_data)
    # print(predictions)