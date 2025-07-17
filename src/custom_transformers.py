from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering operations.
    
    This transformer applies feature engineering operations and ensures they work correctly in cross-validation
    pipelines where transformations need to be applied to each fold.
    
    Operations performed:
    - Splits Cabin column into Deck, Cabin_num, and Side
    - Extracts Group and Passenger_num from PassengerId
    - Calculates FamilySize based on group membership
    - Creates TotalSpent feature from expense columns
    - Converts Cabin_num to numeric format
    - Drops unnecessary columns for modeling (PassengerId, Name)
    """
    
    def __init__(self):
        """Initialize the transformer with no parameters."""
        pass
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Args:
            X: Input data (pandas DataFrame or numpy array)
            y: Target variable (not used, kept for sklearn compatibility)
            
        Returns:
            self: Returns the transformer instance
        """
        # Store column names if X is a DataFrame for later use with numpy arrays
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None
        return self
    
    def transform(self, X):
        """
        Apply feature engineering transformations to the data.
        
        Args:
            X: Input data to transform
            
        Returns:
            X: Transformed data with new features
        """
        # Convert numpy array to DataFrame if needed
        # This handles cases where sklearn converts DataFrames to arrays during pipeline execution
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                # Return as-is if we don't have column names stored
                return X
        else:
            # Create a copy to avoid modifying the original data
            X = X.copy()
        
        # === CABIN FEATURE ENGINEERING ===
        # Split the Cabin column (format: "Deck/Cabin_num/Side") into separate features
        if 'Cabin' in X.columns:
            X[['Deck', 'Cabin_num', 'Side']] = X['Cabin'].str.split('/', expand=True)
            X = X.drop(['Cabin'], axis=1)  # Remove original Cabin column
        
        # === PASSENGER ID FEATURE ENGINEERING ===
        # Extract Group and individual Passenger number from PassengerId (format: "Group_PassengerNum")
        if 'PassengerId' in X.columns:
            X['Group'] = X['PassengerId'].str.split('_').str[0].astype(int)
            X['Passenger_num'] = X['PassengerId'].str.split('_').str[1].astype(int)
        
        # === FAMILY SIZE CALCULATION ===
        # Calculate family size based on how many passengers share the same group
        if 'Group' in X.columns:
            group_sizes = X['Group'].value_counts()  # Count passengers per group
            X['FamilySize'] = X['Group'].map(group_sizes)  # Map group sizes to each passenger
        
        # === TOTAL SPENDING CALCULATION ===
        # Create a feature representing total money spent on board services
        expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        available_expense_cols = [col for col in expense_cols if col in X.columns]
        if available_expense_cols:
            X['TotalSpent'] = X[available_expense_cols].sum(axis=1)
        
        # === DATA TYPE CONVERSIONS ===
        # Convert Cabin_num to numeric format for mathematical operations
        if 'Cabin_num' in X.columns:
            X['Cabin_num'] = pd.to_numeric(X['Cabin_num'], errors='coerce')
        
        # === CLEANUP UNNECESSARY COLUMNS ===
        # Remove columns that are not needed for machine learning modeling
        columns_to_drop = ['PassengerId', 'Passenger_num', 'Name']
        columns_to_drop = [col for col in columns_to_drop if col in X.columns]
        if columns_to_drop:
            X = X.drop(columns_to_drop, axis=1)
        
        return X


class LogicalImputationTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for logical imputation of missing values.
    
    This transformer applies domain-specific logic to fill missing values
    based on relationships in the spaceship dataset:
    - Uses group membership to fill location and demographic information
    - Infers CryoSleep status based on spending patterns
    
    Logical rules:
    - Passengers in the same group likely share location and travel details
    - Passengers with zero expenses are likely in CryoSleep
    - Passengers with positive expenses are likely awake (not in CryoSleep)
    """
    
    def __init__(self):
        """Initialize the transformer with no parameters."""
        pass
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Args:
            X: Input data (pandas DataFrame or numpy array)
            y: Target variable (not used, kept for sklearn compatibility)
            
        Returns:
            self: Returns the transformer instance
        """
        # Store column names if X is a DataFrame for later use with numpy arrays
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None
        return self
    
    def transform(self, X):
        """
        Apply logical imputation to fill missing values.
        
        Args:
            X: Input data to impute
            
        Returns:
            X: Data with missing values filled using logical rules
        """
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                # Return as-is if we don't have column names stored
                return X
        else:
            # Create a copy to avoid modifying the original data
            X = X.copy()
        
        # === GROUP-BASED IMPUTATION ===
        # Fill missing values using information from other passengers in the same group
        if 'Group' in X.columns:
            def fill_location_from_group(row):
                """
                Fill missing location and demographic information using group data.
                
                Logic: Passengers in the same group likely travel together and
                share similar characteristics (cabin location, planet of origin, etc.)
                """
                # Get all passengers in the same group
                group_data = X[X['Group'] == row['Group']]
                
                # Fill location-related columns using most common value in group
                for col in ['Deck', 'Cabin_num', 'Side']:
                    if col in X.columns and pd.isnull(row[col]):
                        mode_val = group_data[col].mode()  # Most frequent value in group
                        if not mode_val.empty:
                            row[col] = mode_val.iloc[0]
                
                # Fill demographic and travel information using group consensus
                for col in ['HomePlanet', 'Destination', 'VIP']:
                    if col in X.columns and pd.isnull(row[col]):
                        mode_val = group_data[col].mode()  # Most frequent value in group
                        if not mode_val.empty:
                            row[col] = mode_val.iloc[0]
                
                return row
            
            # Apply group-based filling to all rows
            X = X.apply(fill_location_from_group, axis=1)
        
        # === CRYOSLEEP LOGICAL IMPUTATION ===
        # Infer CryoSleep status based on spending behavior
        if 'CryoSleep' in X.columns:
            expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
            available_expense_cols = [col for col in expense_cols if col in X.columns]
            
            def infer_cryo(row):
                """
                Infer CryoSleep status based on spending patterns.
                
                Logic:
                - If total expenses = 0, passenger is likely in CryoSleep
                - If total expenses > 0, passenger is likely awake
                - If unable to determine, leave as missing
                """
                if pd.isnull(row['CryoSleep']):
                    # Calculate total expenses (excluding NaN values)
                    total_expense = sum(row[col] for col in available_expense_cols if pd.notna(row[col]))
                    
                    if total_expense == 0:
                        row['CryoSleep'] = True   # No spending = likely in CryoSleep
                    elif total_expense > 0:
                        row['CryoSleep'] = False  # Spending = likely awake
                    else:                    
                        row['CryoSleep'] = np.nan  # Unable to determine, keep missing
                        
                return row['CryoSleep']
            
            # Apply CryoSleep inference to all rows
            X['CryoSleep'] = X.apply(infer_cryo, axis=1)
        
        return X

    
class TransformBooleanToInt(BaseEstimator, TransformerMixin):
    """
    Custom transformer to convert boolean columns to integers.
    
    This transformer converts boolean values (True/False) to integers (1/0)
    to ensure compatibility with machine learning algorithms that expect
    numerical input.
    
    Specifically handles:
    - VIP status (True/False → 1/0)
    - CryoSleep status (True/False → 1/0)
    """
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Args:
            X: Input data (pandas DataFrame or numpy array)
            y: Target variable (not used, kept for sklearn compatibility)
            
        Returns:
            self: Returns the transformer instance
        """
        # Store column names if X is a DataFrame for later use with numpy arrays
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None
        return self
    
    def transform(self, X):
        """
        Convert boolean columns to integer format.
        
        Args:
            X: Input data to transform
            
        Returns:
            X: Data with boolean columns converted to integers
        """
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                # If we don't have stored column names, return as-is 
                # since we can't identify boolean columns
                return X
        else:
            # Create a copy to avoid modifying the original data
            X = X.copy()
        
        # === BOOLEAN TO INTEGER CONVERSION ===
        # Convert specific boolean columns to integer format (True→1, False→0)
        bool_cols = ['VIP', 'CryoSleep']
        for col in bool_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)  # Convert boolean to integer
        
        return X
