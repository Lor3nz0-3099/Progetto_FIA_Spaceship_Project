# Import necessary libraries for data operations and visualization
import pandas as pd                                    
import numpy as np                                     
import matplotlib.pyplot as plt                       
from sklearn.model_selection import train_test_split  
from EDA import EDAAnalyzer                           

class PreliminarOperations:
    """
    Class for handling preliminary data operations in the spaceship ML pipeline.
    
    This class manages the initial data loading, basic preprocessing, and train/test
    splitting operations. It serves as the entry point for the machine learning
    pipeline and prepares raw data for further processing.
    
    Attributes:
        dataset_df: Full dataset loaded from CSV
        X_train: Training features 
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels  
        test_data: Test dataset (if applicable)
    """
    
    def __init__(self):
        """
        Initialize the PreliminarOperations class.
        
        Sets all data attributes to None - they will be populated
        when data loading and processing methods are called.
        """
        self.dataset_df = None  # Will store the full dataset
        self.X_train = None     # Training features
        self.y_train = None     # Training labels
        self.X_val = None       # Validation features  
        self.y_val = None       # Validation labels
        self.test_data = None   # Test data (for final evaluation)

    def load_data(self):
        """
        Load the spaceship dataset from CSV file.
        
        Reads the training data from 'data/train.csv' and displays
        basic information about the dataset shape and first few rows.
        """
        # Load dataset from CSV file
        self.dataset_df = pd.read_csv('data/train.csv')
        
        # Display dataset information for user awareness
        print("Full dataset shape: {}".format(self.dataset_df.shape))
        print(self.dataset_df.head(5))

    def EDA_analysis(self):
        """
        Optionally perform Exploratory Data Analysis (EDA).
        
        Asks the user if they want to run EDA and executes it if requested.
        EDA helps understand data distributions, patterns, and relationships.
        """
        # Ask user if they want to perform EDA
        if input("Do you want to execute EDA? (y/n): ").strip().lower() == 'y':
            # Initialize and run EDA analyzer
            EDAAnalyzer().run_eda(self.dataset_df)

    def plot_label_distribution(self):
        """
        Visualize the distribution of the target variable (Transported).
        
        Creates a bar plot showing how many passengers were transported
        vs not transported. This helps assess class balance in the dataset.
        """
        # Count occurrences of each label value
        plot_df = self.dataset_df.Transported.value_counts()
        
        # Create bar plot
        plot_df.plot(kind="bar")
        plt.title("Label distribution")
        plt.show()

    def split_features_and_labels(self):
        """
        Separate features (input variables) from labels (target variable).
        
        Creates X (features) and y (labels) datasets for machine learning.
        The target variable is 'Transported' (whether passenger was transported).
        """
        label = 'Transported'  # Target variable name
        
        # Create feature matrix (all columns except target)
        self.X = self.dataset_df.drop(label, axis=1)
        
        # Create target vector (convert to integers for compatibility)
        self.y = self.dataset_df[label].astype(int)  # Convert labels to integers
    
    def split_train_val(self):
        """
        Split the dataset into training and validation sets.
        
        Uses stratified splitting to maintain the same proportion of each
        class in both training and validation sets. This ensures balanced
        representation for model training and evaluation.
        """
        # Split data with 80% training, 20% validation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2,        # 20% for validation
            random_state=42,      # For reproducible results
            stratify=self.y       # Maintain class proportions
        )
        
    def get_raw_data_for_pipeline(self):
        """
        Returns raw data that can be used with the ModelFactory pipeline.
        
        This method provides minimal preprocessing - just loading data and
        creating train/validation splits. All feature engineering and
        preprocessing is handled by the pipeline's custom transformers and classical preprocessing techniques.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) - Raw training and validation data
        """
        print("=== LOADING AND PREPARING RAW DATA ===")
        
        # Load the dataset from CSV
        self.load_data()
        
        # Optional EDA for data understanding
        self.EDA_analysis()
        
        # Show target variable distribution
        self.plot_label_distribution()
        
        # Separate features from target variable
        self.split_features_and_labels()
        
        # Create train/validation split
        self.split_train_val()
        
        print("=== RAW DATA PREPARATION COMPLETED ===\n")
        
        # Return raw data for pipeline processing
        return self.X_train, self.X_test, self.y_train, self.y_test


