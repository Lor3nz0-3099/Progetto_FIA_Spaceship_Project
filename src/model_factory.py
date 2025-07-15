# Import necessary libraries and modules
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from custom_transformers import FeatureEngineeringTransformer, LogicalImputationTransformer, TransformBooleanToInt

# Factory class to create and manage different machine learning models and pipelines
class ModelFactory:
    def __init__(self):
        self.kwargs = {} # Optional keyword arguments to pass to models

    def create_model(self, model_type):
        """
        Selects the requested model and its parameter grid or distribution.
        """
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
            return AdaBoostClassifier(random_state=42, **self.kwargs), param_grid

        elif model_type == '3':
            from sklearn.svm import SVC
            param_grid = {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'classifier__kernel': ['linear', 'rbf']
            }
            return SVC(probability=True, random_state=42, verbose=True), param_grid

        elif model_type == '4':
            from xgboost import XGBClassifier
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 4, 6],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
            return XGBClassifier(eval_metric='logloss', random_state=42), param_grid

        elif model_type == '5':
            from catboost import CatBoostClassifier
            param_distributions = {
                'classifier__depth': [4, 5, 6, 7, 8, 9, 10],
                'classifier__iterations': [500, 750, 1000, 1250, 1500],
                'classifier__learning_rate': [0.02, 0.05, 0.07, 0.1, 0.15, 0.2],
                'classifier__l2_leaf_reg': [1, 3, 5, 7, 9, 11],
                'classifier__border_count': [32, 64, 128, 254],
                'classifier__bagging_temperature': [0, 0.5, 1, 2],
                'classifier__random_strength': [0.5, 1, 2, 5],
            }
            return CatBoostClassifier(verbose=0, random_state=42), param_distributions

        elif model_type == '6':
            return self.create_stacking_classifier()

        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

    def create_stacking_classifier(self):
        """
        Builds a StackingClassifier with CatBoost, XGBoost, AdaBoost
        and Logistic Regression as the meta learner.
        """
        from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier

        # Define base learners
        catboost_clf = CatBoostClassifier(
            random_strength=1,
            learning_rate=0.02,
            iterations=1250,
            depth=5,
            border_count=64,
            bagging_temperature=0
        )

        xgboost_clf = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            subsample=1.0,
            random_state=42
        )

        adaboost_clf = AdaBoostClassifier(
            estimator = DecisionTreeClassifier(max_depth=3),
            n_estimators=200,
            learning_rate=1.0,
            random_state=42
        )

        meta_learner = LogisticRegression(max_iter=1000)

        stacked_clf = StackingClassifier(
            estimators=[
                ('catboost', catboost_clf),
                ('xgboost', xgboost_clf),
                ('adaboost', adaboost_clf)
            ],
            final_estimator=meta_learner,
            cv=5,
            passthrough=False,
            n_jobs=-1
        )

        param_grid = {
            'classifier__final_estimator__C': [0.1, 1, 10]
        }

        return stacked_clf, param_grid

    def preprocessing_pipeline(self, numerical_cols, categorical_cols):
        """
        Builds the preprocessing pipeline for numerical and categorical features.
        Includes feature engineering and logical imputations.
        """
        scaler_choice = input("Choose scaling method:\n1. StandardScaler\n2. MinMaxScaler\n3. No scaling\nEnter choice (1/2/3): ").strip()
        scaler = MinMaxScaler() if scaler_choice == '2' else StandardScaler() if scaler_choice == '1' else None

        imputation_numerical_choice = input("Choose imputation strategy:\n1. Mean\n2. KNN\nEnter choice (1/2): ").strip()
        imputation_numerical = SimpleImputer(strategy='mean') if imputation_numerical_choice == '1' else KNNImputer(n_neighbors=5)

        # Numerical preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', imputation_numerical),
            ('scaler', scaler) if scaler else ('passthrough', 'passthrough')
        ])

        # Categorical preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        
        main_preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        # Final preprocessing pipeline with custom steps
        full_preprocessor = Pipeline(steps=[
            ('feature_engineering', FeatureEngineeringTransformer()),
            ('logical_imputation', LogicalImputationTransformer()),
            ('preprocessing', main_preprocessor),
            ('bool2int', TransformBooleanToInt())
        ])

        return full_preprocessor

    def train_and_evaluate_model(self, choice, X_train, y_train, X_val, y_val):
        """
        Builds the full pipeline, runs GridSearchCV or RandomizedSearchCV,
        fits on training data, and evaluates on training and validation sets.
        """

        # Define which features are numerical and which are categorical
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'TotalSpent', 'FamilySize']
        categorical_cols = ['Deck', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']

        # Build preprocessing and model pipelines
        preprocessor = self.preprocessing_pipeline(numerical_cols, categorical_cols)
        model, param_grid = self.create_model(choice)

        # Combine preprocessing and classifier into one pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Use RandomizedSearchCV for CatBoost, GridSearchCV for others
        if choice == '5':
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=50,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
        else:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
        
        # Fit the model with training data using the selected search strategy
        search.fit(X_train, y_train)

        print("Best parameters:", search.best_params_)
        print("Best cross-validated accuracy:", search.best_score_)

        # Make predictions on validation and training sets
        y_pred = search.best_estimator_.predict(X_val)
        y_pred_train = search.best_estimator_.predict(X_train)

        # Print performance metrics
        print("Accuracy on training set:", accuracy_score(y_train, y_pred_train))
        print("Accuracy on validation set:", accuracy_score(y_val, y_pred))
        print("Classification report:\n", classification_report(y_val, y_pred))

        # Plot confusion matrix and ROC curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Confusion matrix heatma
        conf_matrix = confusion_matrix(y_val, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix on Validation Set")

        # ROC curve if model supports probability prediction
        if hasattr(search.best_estimator_.named_steps['classifier'], "predict_proba"):
            y_proba = search.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)

            ax2.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
            ax2.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Receiver Operating Characteristic (ROC)')
            ax2.legend(loc='lower right')
        else:
            # If model doesn't support probabilities, show placeholder text
            ax2.text(0.5, 0.5, 'ROC curve not available\nfor this model',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ROC Curve - Not Available')

        plt.tight_layout()
        plt.show()

        # If model supports feature importances, plot them
        if hasattr(search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
            self.feature_importances(search.best_estimator_, X_train)

        return search.best_estimator_

    def feature_importances(self, model, X_train):
        """
        Computes and plots feature importances for tree-based models.
        """
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_

            try:
                # Extract transformed feature names from preprocessing pipeline
                main_preprocessor = model.named_steps['preprocessor'].named_steps['preprocessing']
                feature_names = []

                # Manually define numerical features (must match your pipeline)
                numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'TotalSpent', 'FamilySize']
                feature_names.extend(numerical_cols)

                # Define categorical features and get their one-hot encoded names
                categorical_cols = ['Deck', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']
                if hasattr(main_preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
                    cat_feature_names = main_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
                else:
                    cat_feature_names = main_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_cols)

                feature_names.extend(cat_feature_names)

            except Exception as e:
                print(f"Could not extract feature names: {e}")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            # Check for mismatch between names and importance scores
            if len(feature_names) != len(importances):
                print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of importances ({len(importances)})")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

             # Group importances by base name (optional, useful for one-hot features)
            grouped_importances = {}
            for i, feature_name in enumerate(feature_names):
                if '_' in feature_name and feature_name not in ['Cabin_num', 'TotalSpent', 'FamilySize']:
                    base_name = '_'.join(feature_name.split('_')[:-1])
                    grouped_importances[base_name] = grouped_importances.get(base_name, 0) + importances[i]
                else:
                    grouped_importances[feature_name] = importances[i]

             # Sort by importance
            grouped_names = list(grouped_importances.keys())
            grouped_values = list(grouped_importances.values())
            indices = sorted(range(len(grouped_values)), key=lambda i: grouped_values[i], reverse=True)

            # Plot grouped importances
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importances (Grouped)")
            plt.bar(range(len(grouped_values)), [grouped_values[i] for i in indices], align='center')
            plt.xticks(range(len(grouped_values)), [grouped_names[i] for i in indices], rotation=45, ha='right')
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.show()
        else:
            print("Feature importances are not available for this model.")
