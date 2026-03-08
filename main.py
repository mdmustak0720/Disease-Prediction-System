# Import Dependencies
import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import numpy as np
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt


class DiseasePrediction:
    # Initialize and Load the Config File
    def __init__(self, model_name=None):
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Verbose
        self.verbose = self.config['verbose']
        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        # Model Definition
        self.model_name = model_name
        # Model Save Path
        self.model_save_path = self.config['model_save_path']

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    # Function to Load Test Dataset
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data sanity
        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    # Features Correlation
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.drop(columns=["prognosis"]).corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(
                len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(
                len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # Model Selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(
                criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(
                n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
        return self.clf

    # ML Model
    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # Training the Model
        classifier = classifier.fit(X_train, y_train)
        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val, y_val)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            clf = load(str(self.model_save_path +
                       saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")
            return None

        if test_data is not None:
            result = clf.predict(test_data)
            return result
        else:
            result = clf.predict(self.test_features)

        # Use new accuracy calculation method
        accuracy_report = self.calculate_disease_accuracy(
            self.test_labels, result)

        # Visualize confusion matrix
        self._visualize_confusion_matrix(
            self.test_labels, result, saved_model_name)

        return accuracy_report

    def calculate_disease_accuracy(self, y_true, y_pred, disease_names=None):
        """Calculate per-disease accuracy and overall metrics"""

        # Overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)

        # Classification report with weighted averages
        report = classification_report(y_true, y_pred, output_dict=True)

        # Per-disease accuracy
        unique_diseases = np.unique(y_true)
        per_disease_accuracy = {}

        for disease in unique_diseases:
            mask = y_true == disease
            if mask.sum() > 0:
                disease_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                per_disease_accuracy[disease] = disease_accuracy

        # Weighted and Macro F1 scores (better for multi-class)
        weighted_f1 = report['weighted avg']['f1-score']
        macro_f1 = report['macro avg']['f1-score']

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DISEASE PREDICTION ACCURACY ANALYSIS")
            print(f"{'='*60}")
            print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")
            print(f"Weighted F1-Score: {weighted_f1:.4f}")
            print(f"Macro F1-Score (avg per disease): {macro_f1:.4f}")
            print(f"\nPer-Disease Accuracy:")
            for disease, acc in sorted(per_disease_accuracy.items(), key=lambda x: x[1]):
                print(f"  {disease}: {acc:.4f}")
            print(f"{'='*60}\n")

        return {
            'overall_accuracy': overall_accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'per_disease_accuracy': per_disease_accuracy,
            'report': report
        }

    def _visualize_confusion_matrix(self, y_true, y_pred, model_name):
        """Visualize confusion matrix for disease predictions"""
        conf_mat = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sn.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=np.unique(y_true),
                   yticklabels=np.unique(y_true))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Disease')
        plt.ylabel('True Disease')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.close()


if __name__ == "__main__":

    # List of models to train
    models = ['decision_tree', 'random_forest', 'gradient_boost', 'mnb']

    # Loop through each model
    for model_name in models:

        print(f"\nTraining model: {model_name}")

        # Create model object
        dp = DiseasePrediction(model_name=model_name)

        # Train the model
        dp.train_model()

        # Test the model
        accuracy_report = dp.make_prediction(saved_model_name=model_name)

        # Print results
        if accuracy_report:
            print(f"\nModel: {model_name}")
            print(
                f"Overall Accuracy: {accuracy_report['overall_accuracy']:.4f}")
            print(f"Weighted F1-Score: {accuracy_report['weighted_f1']:.4f}")
            print(
                f"Per-Disease Accuracy: {accuracy_report['per_disease_accuracy']}")