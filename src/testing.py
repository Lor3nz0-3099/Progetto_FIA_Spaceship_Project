# Import necessary libraries for model testing and evaluation
import pandas as pd                                      
from preliminar_operations import PreliminarOperations  
from model_factory import ModelFactory                  

class Testing:
    """
    Class for testing trained machine learning models on unseen test data.
    
    This class handles the complete testing workflow:
    - Loading test data from CSV files
    - Making predictions using trained models with built-in preprocessing
    - Generating submission files for Kaggle competition
    - Visualizing prediction distributions
    
    The class works with models that contain complete preprocessing pipelines,
    ensuring that the same transformations applied during training are
    automatically applied during testing.
    """
    
    def __init__(self):
        """
        Initialize the Testing class with preliminary operations and model factory.
        
        These components are included for potential future extensions,
        though the current implementation primarily works with pre-trained models.
        """
        self.preliminar_ops = PreliminarOperations()  # For data operations (if needed)
        self.model_factory = ModelFactory()          # For model operations (if needed)

    def load_test_data(self, test_file='data/test.csv'):
        """
        Load test dataset from CSV file.
        
        This method loads the test data that will be used for making predictions.
        The test data typically doesn't contain the target variable (Transported)
        since that's what we're trying to predict.
        
        Args:
            test_file (str): Path to the test CSV file (default: 'data/test.csv')
            
        Returns:
            pandas.DataFrame: Test dataset ready for prediction
        """
        # Load test dataset from CSV file
        test_df = pd.read_csv(test_file)
        
        # Display basic information about the test dataset
        print("Test dataset shape: {}".format(test_df.shape))
        print(test_df.head(5))  # Show first 5 rows for inspection
        
        return test_df
    
    def test_model(self, model):
        """
        Test the trained model on unseen test data and generate predictions.
        
        This method performs the complete testing workflow:
        1. Loads test data
        2. Extracts passenger IDs for submission file
        3. Makes predictions using the model's built-in preprocessing pipeline
        4. Saves predictions to submission file
        5. Visualizes prediction distribution
        
        The model is expected to be a complete pipeline that includes
        all necessary preprocessing steps (feature engineering, imputation,
        scaling, encoding) so no additional preprocessing is needed here.
        
        Args:
            model: Trained sklearn Pipeline containing preprocessor + classifier
            
        Returns:
            pandas.DataFrame: DataFrame containing PassengerId and predictions
        """
        # === LOAD TEST DATA ===
        test_df = self.load_test_data()
        
        # === EXTRACT PASSENGER IDS ===
        # Store PassengerId for the final submission file
        # This is crucial for competition submissions where each prediction
        # must be matched to the correct passenger
        passenger_ids = test_df['PassengerId'].copy()
        
        # === MAKE PREDICTIONS ===
        # Use the trained model to make predictions on test data
        # The model.predict() automatically applies all preprocessing steps:
        # 1. Feature engineering (cabin splitting, group extraction, etc.)
        # 2. Logical imputation (group-based filling, CryoSleep inference)
        # 3. Classical preprocessing (scaling, encoding, imputation)
        # 4. Final prediction using the trained classifier
        predictions = model.predict(test_df).astype(bool)  # Convert to boolean as expected
        
        # === SAVE SUBMISSION FILE ===
        # Create submission DataFrame in the required format
        submission_df = pd.DataFrame({
            'PassengerId': passenger_ids,    # Unique identifier for each passenger
            'Transported': predictions       # Model predictions (True/False)
        })
        
        # Save to CSV file for competition submission
        submission_df.to_csv('data/submission_new.csv', index=False)
        print("Predictions saved to submission_new.csv")

        # === PREPARE RETURN DATA ===
        # Create predictions DataFrame for further analysis
        predictions_df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Transported': predictions
        })

        print("Predictions DataFrame shape: {}".format(predictions_df.shape))

        # === VISUALIZE PREDICTIONS ===
        # Plot the prediction distribution to understand model behavior
        self.plot_predictions(predictions_df)
        
        return predictions_df
    
    def plot_predictions(self, predictions):
        """
        Visualize the distribution of model predictions.
        
        Creates a count plot showing how many passengers the model predicts
        will be transported vs not transported. This helps understand:
        - Model prediction balance
        - Whether the model shows reasonable prediction patterns
        - Potential bias toward one class
        
        Args:
            predictions (pandas.DataFrame): DataFrame containing predictions
                                          with columns ['PassengerId', 'Transported']
        """
        # Import visualization libraries
        import matplotlib.pyplot as plt
        import seaborn as sns

        # === CREATE PREDICTION DISTRIBUTION PLOT ===
        plt.figure(figsize=(8, 5))
        
        # Create count plot showing distribution of predictions
        sns.countplot(x='Transported', data=predictions, palette='Set1')
        
        # Customize plot appearance
        plt.title('Distribution of Transported Predictions', fontsize=14)
        plt.xlabel('Transported', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)  # Keep labels horizontal
        plt.tight_layout()
        
        # Display the plot
        plt.show()
        
        # === PRINT PREDICTION STATISTICS ===
        # Calculate and display prediction statistics
        total_predictions = len(predictions)
        transported_count = predictions['Transported'].sum()
        not_transported_count = total_predictions - transported_count
        
        print(f"\n=== PREDICTION STATISTICS ===")
        print(f"Total predictions: {total_predictions}")
        print(f"Predicted as Transported: {transported_count} ({transported_count/total_predictions:.2%})")
        print(f"Predicted as Not Transported: {not_transported_count} ({not_transported_count/total_predictions:.2%})")