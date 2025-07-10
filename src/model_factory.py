from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from custom_transformers import FeatureEngineeringTransformer, LogicalImputationTransformer, TransformBooleanToInt

class ModelFactory:
    def __init__(self):
        self.kwargs = {}

    def create_model(self, model_type):
        """
        Selects the requested model and its parameter grid based on the given model type.
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
            param_grid = {
                'classifier__iterations': [100, 200],
                'classifier__depth': [3, 4, 5],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__l2_leaf_reg': [1, 3, 5]
            }
            return CatBoostClassifier(verbose=0, random_state=42), param_grid

        elif model_type == '6':
            # Use the dedicated stacking method
            return self.create_stacking_classifier()

        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

    def create_stacking_classifier(self):
        """
        Builds a StackingClassifier combining CatBoost, XGBoost, AdaBoost
        with Logistic Regression as the meta learner.
        """
        from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier

        # === Base learners ===
        catboost_clf = CatBoostClassifier(
            depth=6,
            iterations=200,
            l2_leaf_reg=3,
            learning_rate=0.1,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            verbose=0,
            random_state=42
        )

        xgboost_clf = XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.2,
            subsample=0.9,
            eval_metric='logloss',
            random_state=42
        )

        adaboost_clf = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=1.0,
            random_state=42
        )

        # Meta learner
        meta_learner = LogisticRegression(max_iter=1000)

        # Stacking classifier
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

        # Optional param grid for tuning the meta learner
        param_grid = {
           
            'classifier__final_estimator__C': [0.1, 1, 10]
        }

        return stacked_clf, param_grid

    def preprocessing_pipeline(self, numerical_cols, categorical_cols):
        """
        Builds the preprocessing pipeline for numerical and categorical features.
        Includes feature engineering, logical imputations, and regular preprocessing based on user input.
        """
        # Choose scaler
        scaler_choice = input("Choose scaling method:\n1. StandardScaler\n2. MinMaxScaler\n3. No scaling\nEnter choice (1/2/3): ").strip()
        if scaler_choice not in ['1', '2', '3']:
            print("Invalid choice, defaulting to StandardScaler.")
            scaler_choice = '1'
        scaler = MinMaxScaler() if scaler_choice == '2' else StandardScaler() if scaler_choice == '1' else None

        # Choose imputation strategy for numerical data
        imputation_numerical_choice = input("Choose imputation strategy:\n1. Mean\n2. KNN\nEnter choice (1/2): ").strip()
        if imputation_numerical_choice not in ['1', '2']:
            print("Invalid choice, defaulting to mean imputation.")
            imputation_numerical_choice = '1'
        imputation_numerical = SimpleImputer(strategy='mean') if imputation_numerical_choice == '1' else KNNImputer(n_neighbors=5)

        # Numerical pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', imputation_numerical),
            ('scaler', scaler) if scaler else ('passthrough', 'passthrough')
        ])

        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Main preprocessing pipeline with logical imputations
        main_preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        # Complete pipeline: feature engineering -> logical imputations -> standard preprocessing
        full_preprocessor = Pipeline(steps=[
            ('feature_engineering', FeatureEngineeringTransformer()),
            ('logical_imputation', LogicalImputationTransformer()),
            ('preprocessing', main_preprocessor),
            ('bool2int', TransformBooleanToInt())
        ])

        return full_preprocessor

    def train_and_evaluate_model(self, choice, X_train, y_train, X_val, y_val):
        """
        Builds the full pipeline, runs GridSearchCV, fits on training data,
        and evaluates on training and validation sets.
        Expects raw data with all original columns (including PassengerId, Cabin, etc.)
        """
        # Feature lists (will be created after feature engineering)
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'TotalSpent', 'FamilySize']
        categorical_cols = ['Deck', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']

        # Build preprocessing pipeline
        preprocessor = self.preprocessing_pipeline(numerical_cols, categorical_cols)

        # Get the model and its param grid
        model, param_grid = self.create_model(choice)

        # Full pipeline: preprocessor + model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # 5-fold stratified CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Best results
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validated accuracy:", grid_search.best_score_)

        # Predictions and evaluation
        y_pred = grid_search.predict(X_val)
        y_pred_train = grid_search.predict(X_train)

        print("Accuracy on training set:", accuracy_score(y_train, y_pred_train))
        print("Accuracy on validation set:", accuracy_score(y_val, y_pred))
        print("Classification report:\n", classification_report(y_val, y_pred))

        # Create subplots for confusion matrix and ROC curve side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix plot
        conf_matrix = confusion_matrix(y_val, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix on Validation Set")

        # ROC curve
        if hasattr(grid_search.best_estimator_.named_steps['classifier'], "predict_proba"):
            y_proba = grid_search.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)

            ax2.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
            ax2.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Receiver Operating Characteristic (ROC)')
            ax2.legend(loc='lower right')
        else:
            ax2.text(0.5, 0.5, 'ROC curve not available\nfor this model', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ROC Curve - Not Available')
        
        plt.tight_layout()
        plt.show()

        # Feature importances for tree-based models
        if hasattr(grid_search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
            self.feature_importances(grid_search.best_estimator_, X_train)

        return grid_search.best_estimator_
    
    def feature_importances(self, model, X_train):
        """
        Computes and plots feature importances for tree-based models.
        """
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            
            # Get feature names from the preprocessor
            try:
                # Access the nested preprocessor inside our pipeline
                main_preprocessor = model.named_steps['preprocessor'].named_steps['preprocessing']
                feature_names = []
                
                # Get numerical feature names
                numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'TotalSpent', 'FamilySize']
                feature_names.extend(numerical_cols)
                
                # Get categorical feature names after OneHotEncoding
                categorical_cols = ['Deck', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']
                if hasattr(main_preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
                    cat_feature_names = main_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
                    feature_names.extend(cat_feature_names)
                else:
                    # Fallback for older scikit-learn versions
                    cat_feature_names = main_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_cols)
                    feature_names.extend(cat_feature_names)
                    
            except Exception as e:
                print(f"Could not extract feature names: {e}")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
            # Ensure we have the right number of feature names
            if len(feature_names) != len(importances):
                print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of importances ({len(importances)})")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
            # Group one-hot encoded features by their base name
            grouped_importances = {}
            for i, feature_name in enumerate(feature_names):
                # Check if it's a one-hot encoded feature (contains underscore)
                if '_' in feature_name and feature_name not in ['Cabin_num', 'TotalSpent', 'FamilySize']:
                    # Extract base name (everything before the last underscore)
                    base_name = '_'.join(feature_name.split('_')[:-1])
                    if base_name in grouped_importances:
                        grouped_importances[base_name] += importances[i]
                    else:
                        grouped_importances[base_name] = importances[i]
                else:
                    # Keep numerical features as they are
                    grouped_importances[feature_name] = importances[i]
            
            # Convert to lists for plotting
            grouped_names = list(grouped_importances.keys())
            grouped_values = list(grouped_importances.values())
            
            # Sort by importance
            indices = sorted(range(len(grouped_values)), key=lambda i: grouped_values[i], reverse=True)

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

