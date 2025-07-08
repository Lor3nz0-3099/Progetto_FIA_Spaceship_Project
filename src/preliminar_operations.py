import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from EDA import run_eda

class PreliminarOperations:
    def __init__(self):
        self.dataset_df = None

    def load_data(self):
        self.dataset_df = pd.read_csv('data/train.csv')
        print("Full dataset shape: {}".format(self.dataset_df.shape))
        print(self.dataset_df.head(5))

    def EDA_analysis(self):
        if input("Vuoi eseguire l'EDA? (y/n): ").strip().lower() == 'y':
            run_eda(self.dataset_df)

    def plot_label_distribution(self):
        plot_df = self.dataset_df.Transported.value_counts()
        plot_df.plot(kind="bar")
        plt.title("Label distribution")
        plt.show()

    def split_columns(self):
        self.dataset_df[['Deck', 'Cabin_num', 'Side']] = self.dataset_df['Cabin'].str.split('/', expand=True) 
        self.dataset_df = self.dataset_df.drop(['Name', 'Cabin'], axis=1)
        print(self.dataset_df.head(5))

    def feature_engineering(self):
        # Extract Group and Passenger_num from PassengerId
        self.dataset_df['Group'] = self.dataset_df['PassengerId'].str.split('_').str[0].astype(int)
        self.dataset_df['Passenger_num'] = self.dataset_df['PassengerId'].str.split('_').str[1].astype(int)

        # Calculate FamilySize
        group_sizes = self.dataset_df['Group'].value_counts()
        self.dataset_df['FamilySize'] = self.dataset_df['Group'].map(group_sizes)

        # Total spent on board
        expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.dataset_df['TotalSpent'] = self.dataset_df[expense_cols].sum(axis=1)
        print(f"Total spent and Groups on board calculated. Sample:\n{self.dataset_df[['Group', 'TotalSpent', 'Passenger_num']].head(5)}")

        # Convert Cabin_num to numeric, handling errors
        self.dataset_df['Cabin_num'] = pd.to_numeric(self.dataset_df['Cabin_num'], errors='coerce')

    def fill_missing_values(self):
        missing_values = self.dataset_df.isnull().sum()
        print("Missing values in each column:")
        print(missing_values[missing_values > 0])

        # Fill missing Deck/Side/Cabin_num based on shared Group information
        # If any of these are missing, we will fill them based on the mode of the group
        # If all are present, we will not fill them
        def fill_location_from_surname(row):
            if pd.isnull(row['Deck']) or pd.isnull(row['Cabin_num']) or pd.isnull(row['Side']):
                group = self.dataset_df[self.dataset_df['Group'] == row['Group']]
                for col in ['Deck', 'Cabin_num', 'Side']:
                    if pd.isnull(row[col]):
                        mode_val = group[col].mode()
                        if not mode_val.empty:
                            row[col] = mode_val.iloc[0] 
            elif pd.isnull(row['HomePlanet']):
                group = self.dataset_df[self.dataset_df['Group'] == row['Group']]
                mode_homeplanet = group['HomePlanet'].mode()
                if not mode_homeplanet.empty:
                    row['HomePlanet'] = mode_homeplanet.iloc[0]
            elif pd.isnull(row['Destination']):
                group = self.dataset_df[self.dataset_df['Group'] == row['Group']]
                mode_destination = group['Destination'].mode()
                if not mode_destination.empty:
                    row['Destination'] = mode_destination.iloc[0] 
            elif pd.isnull(row['VIP']):
                group = self.dataset_df[self.dataset_df['Group'] == row['Group']]
                mode_vip = group['VIP'].mode()
                if not mode_vip.empty:
                    row['VIP'] = mode_vip.iloc[0] 
            return row
        
        self.dataset_df = self.dataset_df.apply(fill_location_from_surname, axis=1)

        # Fill missing CryoSleep using expenses
        expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        def infer_cryo(row):
            if pd.isnull(row['CryoSleep']):
                total_expense = sum(row[col] for col in expense_cols if pd.notna(row[col]))
                return total_expense == 0
            return row['CryoSleep']
        
        self.dataset_df['CryoSleep'] = self.dataset_df.apply(infer_cryo, axis=1)

        # Drop unused or now-unnecessary columns
        self.dataset_df = self.dataset_df.dropna(subset=['Deck', 'Cabin_num', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP'])
        self.dataset_df = self.dataset_df.drop(['PassengerId'], axis=1)
    
    def split_features_and_labels(self):
        label = 'Transported'
        X = self.dataset_df.drop(label, axis=1)
        y = self.dataset_df[label].astype(int)  # Convert labels to integers
        
        # Convert booleans to integers
        X['VIP'] = X['VIP'].astype(int)
        X['CryoSleep'] = X['CryoSleep'].astype(int)
        
        return X, y
    
    def split_train_val(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def preliminar_operations(self):
        self.load_data()
        self.EDA_analysis()
        self.plot_label_distribution()
        self.split_columns()
        self.feature_engineering()
        self.fill_missing_values()
        X, y = self.split_features_and_labels()
        X_train, X_val, y_train, y_val = self.split_train_val(X, y)
        return X_train, X_val, y_train, y_val


