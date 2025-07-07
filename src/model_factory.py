from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

class ModelFactory:
    def __init__(self):
        self.kwargs = {}
    def create_model(self, model_type):
        if model_type == '1':
            from sklearn.ensemble import RandomForestClassifier
            param_grid = {
                'classifier__n_estimators': [100, 200, 250],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [5, 10, 15]
            }
            return RandomForestClassifier(random_state=42, **self.kwargs), param_grid
        
        elif model_type == '2':
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier 
            base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42)
            self.kwargs['estimator'] = base_estimator
            self.kwargs['random_state'] = 42
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.5, 1.0, 1.5]
            }
            return AdaBoostClassifier(**self.kwargs), param_grid
        
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        
    def train_and_evaluate_model(self, choice, X_train, y_train, X_val, y_val):
        # Identify numerical and categorical features
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        categorical_cols = ['Deck', 'Cabin_num', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']

        # Define preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
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
    

    









