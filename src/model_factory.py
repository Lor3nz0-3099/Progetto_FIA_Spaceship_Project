from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelFactory:
    def __init__(self):
        self.kwargs = {}
    def create_model(self, model_type):
        if model_type == '1':
            from sklearn.ensemble import RandomForestClassifier
            param_grid = {
                'classifier__n_estimators': [200, 300, 400],
                'classifier__max_depth': [5, 10, 15],
                'classifier__max_features': ['sqrt', 'log2'],
                'classifier__min_samples_split': [5, 10],
                'classifier__min_samples_leaf': [2, 4, 6],
                'classifier__bootstrap': [True]
            }
            return RandomForestClassifier(random_state=42, **self.kwargs), param_grid
        
        elif model_type == '2':
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier 
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.5, 1.0],
                'classifier__estimator': [
                    DecisionTreeClassifier(max_depth=1),
                    DecisionTreeClassifier(max_depth=2),
                    DecisionTreeClassifier(max_depth=3)
                ]
            }
            model = AdaBoostClassifier(random_state=42, **self.kwargs)
            return AdaBoostClassifier(), param_grid
        
        elif model_type == '3':
            from sklearn.svm import SVC
            param_grid = {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'classifier__kernel': ['linear', 'rbf']
            }
            model = SVC(probability=True, random_state=42, verbose=True)
            return model, param_grid
        
        elif model_type == '4':
            from xgboost import XGBClassifier
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 4, 6],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
            model = XGBClassifier(eval_metric='logloss', random_state=42)
            return model, param_grid
        elif model_type == '5':
            from catboost import CatBoostClassifier
            param_grid = {
                'classifier__iterations': [100, 200],
                'classifier__depth': [3, 4, 5],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__l2_leaf_reg': [1, 3, 5]
            }
            model = CatBoostClassifier(verbose=0, random_state=42)
            return model, param_grid
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
    
    def preprocessing_pipeline(self, numerical_cols, categorical_cols):
        
        # Define numerical transformer
        # Choose scaling method
        scaler_choice = input("Choose scaling method:\n1. StandardScaler\n2. MinMaxScaler\n3. No scaling\nEnter choice (1/2/3): ").strip()
        if scaler_choice not in ['1', '2', '3']:
            print("Invalid choice, defaulting to StandardScaler.")
            scaler_choice = '1'
        scaler = MinMaxScaler() if scaler_choice == '2' else StandardScaler() if scaler_choice == '1' else None
        
        # Choose imputation strategy
        imputation_numerical_choice = input("Choose imputation strategy:\n1. Mean\n2. knn\nEnter choice (1/2): ").strip()
        if imputation_numerical_choice not in ['1', '2']:
            print("Invalid choice, defaulting to mean imputation.")
            imputation_numerical_choice = '1'
        imputation_numerical = SimpleImputer(strategy='mean') if imputation_numerical_choice == '1' else KNNImputer(n_neighbors=5)
        

        numerical_transformer = Pipeline(steps=[
            ('imputer', imputation_numerical),
            ('scaler', scaler) if scaler else ('passthrough', 'passthrough')
        ])

        # Define categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers into a preprocessor
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
        return preprocessor
        
    def train_and_evaluate_model(self, choice, X_train, y_train, X_val, y_val):
        # Identify numerical and categorical features
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'TotalSpent', 'FamilySize']
        categorical_cols = ['Deck', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']

        preprocessor = self.preprocessing_pipeline(numerical_cols, categorical_cols)
        
        model, param_grid = self.create_model(choice)
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Show best parameters and performance
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validated accuracy:", grid_search.best_score_)

        # Predict on full training data for evaluation (optional)
        y_pred = grid_search.predict(X_val)
        y_pred_train = grid_search.predict(X_train)
        print("Accuracy on training set:", accuracy_score(y_train, y_pred_train))
        print("Accuracy on validation set:", accuracy_score(y_val, y_pred))
        print("Classification report:\n", classification_report(y_val, y_pred))        # Confusion matrix
        conf_matrix = confusion_matrix(y_val, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix on Validation Set")
        plt.show()
        
        return grid_search.best_estimator_
    

    









