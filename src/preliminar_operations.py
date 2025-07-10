import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from EDA import EDAAnalyzer

class PreliminarOperations:
    def __init__(self):
        self.dataset_df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.test_data = None

    def load_data(self):
        self.dataset_df = pd.read_csv('data/train.csv')
        print("Full dataset shape: {}".format(self.dataset_df.shape))
        print(self.dataset_df.head(5))

    def EDA_analysis(self):
        if input("Do you want to execute EDA? (y/n): ").strip().lower() == 'y':
            EDAAnalyzer().run_eda(self.dataset_df)

    def plot_label_distribution(self):
        plot_df = self.dataset_df.Transported.value_counts()
        plot_df.plot(kind="bar")
        plt.title("Label distribution")
        plt.show()

    def split_columns(self):
        self.dataset_df[['Deck', 'Cabin_num', 'Side']] = self.dataset_df['Cabin'].str.split('/', expand=True) 
        self.dataset_df = self.dataset_df.drop(['Name', 'Cabin'], axis=1)
        print(self.dataset_df.head(5))

    def split_features_and_labels(self):
        label = 'Transported'
        self.X = self.dataset_df.drop(label, axis=1)
        self.y = self.dataset_df[label].astype(int)  # Convert labels to integers
    
    def split_train_val(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
    def get_raw_data_for_pipeline(self):
        """
        Returns raw data that can be used with the new ModelFactory pipeline.
        Only performs basic data loading and train/test split without preprocessing.
        """
        # Load the data
        self.load_data()
        
        # Show EDA if requested
        self.EDA_analysis()
        self.plot_label_distribution()
        
        # Split features and labels
        self.split_features_and_labels()
        
        # Split train/validation
        self.split_train_val()
        
        return self.X_train, self.X_test, self.y_train, self.y_test


