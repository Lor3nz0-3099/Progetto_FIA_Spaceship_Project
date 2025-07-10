# codice che valuta il test.csv con il best model trovato e salvato su joblib
import joblib
import pandas as pd
from preliminar_operations import PreliminarOperations
from model_factory import ModelFactory

class Testing:
    def __init__(self):
        self.preliminar_ops = PreliminarOperations()
        self.model_factory = ModelFactory()

    def load_test_data(self, test_file='data/test.csv'):
        test_df = pd.read_csv(test_file)
        print("Test dataset shape: {}".format(test_df.shape))
        print(test_df.head(5))
        return test_df
    
    def test_model(self, model, model_type):
        """
        Test the model using the new pipeline approach.
        The model already contains all preprocessing steps including logical imputations.
        """
        test_df = self.load_test_data()
        
        # Extract PassengerId for submission
        passenger_ids = test_df['PassengerId'].copy()
        
        # Make predictions using the model's built-in preprocessing pipeline
        # The model already contains the complete preprocessing pipeline
        predictions = model.predict(test_df).astype(bool)
        
        # Save predictions to CSV
        submission_df = pd.DataFrame({
            'PassengerId': passenger_ids, 
            'Transported': predictions
        })
        submission_df.to_csv('data/submission_new.csv', index=False)
        print("Predictions saved to submission_new.csv")

        # Return predictions as a DataFrame
        predictions_df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Transported': predictions
        })

        print("Predictions DataFrame shape: {}".format(predictions_df.shape))

        # Plot the predictions to visualize the distribution of Transported values
        self.plot_predictions(predictions_df)
        return predictions_df
    
    def plot_predictions(self, predictions):
        """
        Plot the predictions to visualize the distribution of Transported values.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 5))
        sns.countplot(x='Transported', data=predictions, palette='Set1')
        plt.title('Distribution of Transported Predictions', fontsize=14)
        plt.xlabel('Transported', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()