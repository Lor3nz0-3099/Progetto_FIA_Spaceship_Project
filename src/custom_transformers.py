from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personalizzato per il feature engineering.
    Applica le stesse operazioni di preliminar_operations ma su ogni fold.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Store column names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None
        return self
    
    def transform(self, X):
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                # Return as-is if we don't have column names
                return X
        else:
            X = X.copy()
        
         # Split 'Cabin' into three separate features: Deck, Cabin_num, Side
        if 'Cabin' in X.columns:
            X[['Deck', 'Cabin_num', 'Side']] = X['Cabin'].str.split('/', expand=True)
            X = X.drop(['Cabin'], axis=1)
        
        # Extract Group and Passenger_num from PassengerId
        if 'PassengerId' in X.columns:
            X['Group'] = X['PassengerId'].str.split('_').str[0].astype(int)
            X['Passenger_num'] = X['PassengerId'].str.split('_').str[1].astype(int)
        
        # Calculate FamilySize
        if 'Group' in X.columns:
            group_sizes = X['Group'].value_counts()
            X['FamilySize'] = X['Group'].map(group_sizes)
        
        # Total spent on board
        expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        available_expense_cols = [col for col in expense_cols if col in X.columns]
        if available_expense_cols:
            X['TotalSpent'] = X[available_expense_cols].sum(axis=1)
        
        # Convert Cabin_num to numeric
        if 'Cabin_num' in X.columns:
            X['Cabin_num'] = pd.to_numeric(X['Cabin_num'], errors='coerce')
        
        # Drop columns we don't need for modeling
        columns_to_drop = ['PassengerId', 'Passenger_num', 'Name']
        columns_to_drop = [col for col in columns_to_drop if col in X.columns]
        if columns_to_drop:
            X = X.drop(columns_to_drop, axis=1)
        
        return X


class LogicalImputationTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personalizzato per le imputazioni logiche.
    Applica le stesse logiche di preliminar_operations ma su ogni fold.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Store column names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None
        return self
    
    def transform(self, X):
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                # Return as-is if we don't have column names
                return X
        else:
            X = X.copy()
        
        # Fill missing location information (Deck/Side/Cabin_num) based on Group
        if 'Group' in X.columns:
            def fill_location_from_group(row):
                group_data = X[X['Group'] == row['Group']]
                
                # Fill location-related columns
                for col in ['Deck', 'Cabin_num', 'Side']:
                    if col in X.columns and pd.isnull(row[col]):
                        mode_val = group_data[col].mode()
                        if not mode_val.empty:
                            row[col] = mode_val.iloc[0]
                
                # Fill other group-related information
                for col in ['HomePlanet', 'Destination', 'VIP']:
                    if col in X.columns and pd.isnull(row[col]):
                        mode_val = group_data[col].mode()
                        if not mode_val.empty:
                            row[col] = mode_val.iloc[0]
                
                return row
            
            X = X.apply(fill_location_from_group, axis=1)
        
        # Infer CryoSleep: if a passenger spent absolutely nothing, assume they were asleep
        if 'CryoSleep' in X.columns:
            expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
            available_expense_cols = [col for col in expense_cols if col in X.columns]
            
            def infer_cryo(row):
                if pd.isnull(row['CryoSleep']):
                    total_expense = sum(row[col] for col in available_expense_cols if pd.notna(row[col]))
                    if total_expense == 0:
                        row['CryoSleep'] = True
                    elif total_expense > 0:
                        row['CryoSleep'] = False
                    else:                    
                        row['CryoSleep'] = np.nan  # Fallback if total_expense is NaN or negative
                return row['CryoSleep']
            
            X['CryoSleep'] = X.apply(infer_cryo, axis=1)
        
        return X
    
class TransformBooleanToInt(BaseEstimator, TransformerMixin):
    """
    Transformer to convert boolean columns to integers (0/1).
    """
    
    def fit(self, X, y=None):
        # Store column names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None
        return self
    
    def transform(self, X):
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                # If we don't have stored column names, return as-is since we can't identify boolean columns
                return X
        else:
            X = X.copy()
        
        bool_cols = ['VIP', 'CryoSleep']
        for col in bool_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        return X
