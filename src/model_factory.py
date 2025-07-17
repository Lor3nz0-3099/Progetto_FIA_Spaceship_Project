from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold  
from sklearn.pipeline import Pipeline                                                   
from sklearn.impute import SimpleImputer, KNNImputer                                   
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder          
from sklearn.compose import ColumnTransformer                                          
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc  
import seaborn as sns                                                                   
import matplotlib.pyplot as plt                                                       
from custom_transformers import FeatureEngineeringTransformer, LogicalImputationTransformer, TransformBooleanToInt  


class ModelFactory:
    """
    Factory class for creating, training, and evaluating machine learning models.
    
    This class provides a unified interface for working with different ML algorithms
    and handles the complete pipeline from data preprocessing to model evaluation.
    
    Supported models:
    - Random Forest Classifier
    - AdaBoost Classifier
    - Support Vector Machine (SVM)
    - XGBoost Classifier
    - CatBoost Classifier
    - Stacked Classifier (ensemble of multiple models)
    """
    
    def __init__(self):
        """
        Initialize the ModelFactory with empty keyword arguments.
        
        The kwargs dictionary can be used to pass additional parameters
        to model constructors if needed.
        """
        self.kwargs = {}

    def create_model(self, model_type):
        """
        Factory method to create the requested model and its parameter grid.
        
        Args:
            model_type (str): String identifier for the model type ('1'-'6')
                             '1': Random Forest, '2': AdaBoost, '3': SVM,
                             '4': XGBoost, '5': CatBoost, '6': Stacked Classifier
        
        Returns:
            tuple: (model_instance, parameter_grid) for hyperparameter tuning
        """
        if model_type == '1':
            # === RANDOM FOREST CLASSIFIER ===
            from sklearn.ensemble import RandomForestClassifier
            # Define hyperparameter grid for Random Forest tuning
            param_grid = {
                'classifier__n_estimators': [200, 300, 400],           # Number of trees in forest
                'classifier__max_depth': [5, 10, 15],                  # Maximum depth of trees
                'classifier__max_features': ['sqrt', 'log2'],          # Number of features to consider for each tree
                'classifier__min_samples_split': [5, 10],              # Minimum samples required to split internal node
                'classifier__min_samples_leaf': [2, 4, 6],             # Minimum samples required at leaf node
                'classifier__bootstrap': [True]                        # Whether to use bootstrap sampling
            }
            return RandomForestClassifier(random_state=42, **self.kwargs), param_grid

        elif model_type == '2':
            # === ADABOOST CLASSIFIER ===
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            # Define hyperparameter grid for AdaBoost tuning
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],           # Number of boosting stages
                'classifier__learning_rate': [0.5, 1.0],               # Learning rate shrinks contribution of each classifier
                'classifier__estimator': [                             # Base estimators with different depths
                    DecisionTreeClassifier(max_depth=1),               # Stumps (single split)
                    DecisionTreeClassifier(max_depth=2),               # Shallow trees
                    DecisionTreeClassifier(max_depth=3)                # Slightly deeper trees
                ]
            }
            return AdaBoostClassifier(random_state=42, **self.kwargs), param_grid

        elif model_type == '3':
            # === SUPPORT VECTOR MACHINE (SVM) ===
            from sklearn.svm import SVC
            # Define hyperparameter grid for SVM tuning
            param_grid = {
                'classifier__C': [0.1, 1, 10, 100],                   # Regularization parameter
                'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1], # Kernel coefficient
                'classifier__kernel': ['linear', 'rbf']               # Kernel type for algorithm
            }
            return SVC(probability=True, random_state=42, verbose=True), param_grid

        elif model_type == '4':
            # === XGBOOST CLASSIFIER ===
            from xgboost import XGBClassifier
            # Define hyperparameter grid for XGBoost tuning
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],          # Number of gradient boosted trees
                'classifier__max_depth': [3, 4, 6],                   # Maximum tree depth
                'classifier__learning_rate': [0.05, 0.1, 0.2],       # Boosting learning rate
                'classifier__subsample': [0.8, 0.9, 1.0]             # Subsample ratio of training instances
            }
            return XGBClassifier(eval_metric='logloss', random_state=42), param_grid

        elif model_type == '5':
            # === CATBOOST CLASSIFIER ===
            from catboost import CatBoostClassifier
            # Define parameter distributions for RandomizedSearchCV (CatBoost has many parameters)
            param_distributions = {
                'classifier__depth': [4, 5, 6, 7, 8, 9, 10],                    # Tree depth
                'classifier__iterations': [500, 750, 1000, 1250, 1500],         # Number of boosting iterations
                'classifier__learning_rate': [0.02, 0.05, 0.07, 0.1, 0.15, 0.2], # Learning rate
                'classifier__l2_leaf_reg': [1, 3, 5, 7, 9, 11],                 # L2 regularization coefficient
                'classifier__border_count': [32, 64, 128, 254],                  # Number of splits for numerical features
                'classifier__bagging_temperature': [0, 0.5, 1, 2],              # Controls intensity of Bayesian bagging
                'classifier__random_strength': [0.5, 1, 2, 5],                  # Amount of randomness for scoring splits
            }
            return CatBoostClassifier(verbose=0, random_state=42), param_distributions

        elif model_type == '6':
            # === STACKED CLASSIFIER (ENSEMBLE) ===
            return self.create_stacking_classifier()

        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

    def create_stacking_classifier(self):
        """
        Creates a Stacking Classifier ensemble with multiple base models.
        
        The stacking approach uses:
        - Base models: CatBoost, XGBoost, AdaBoost (diverse algorithms)
        - Meta-learner: Logistic Regression (combines base model predictions)
        
        Returns:
            tuple: (stacked_classifier, parameter_grid) for hyperparameter tuning
        """
        from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier

        # === BASE MODEL 1: CATBOOST ===
        # Configure CatBoost with optimized parameters
        catboost_clf = CatBoostClassifier(
            random_strength=1,         # Randomness for scoring splits
            learning_rate=0.02,        # Conservative learning rate
            iterations=1250,           # High number of iterations
            depth=5,                   # Moderate tree depth
            border_count=64,           # Number of splits for numerical features
            bagging_temperature=0      # No Bayesian bagging
        )

        # === BASE MODEL 2: XGBOOST ===
        # Configure XGBoost with balanced parameters
        xgboost_clf = XGBClassifier(
            max_depth=4,               # Moderate tree depth
            n_estimators=200,          # Number of trees
            learning_rate=0.1,         # Standard learning rate
            subsample=1.0,             # Use all training instances
            random_state=42            # For reproducibility
        )

        # === BASE MODEL 3: ADABOOST ===
        # Configure AdaBoost with decision tree base estimators
        adaboost_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),  # Base estimator
            n_estimators=200,          # Number of boosting stages
            learning_rate=1.0,         # Learning rate
            random_state=42            # For reproducibility
        )

        # === META-LEARNER: LOGISTIC REGRESSION ===
        # Simple linear model to combine base model predictions
        meta_learner = LogisticRegression(max_iter=1000)

        # === CREATE STACKING CLASSIFIER ===
        stacked_clf = StackingClassifier(
            estimators=[
                ('catboost', catboost_clf),
                ('xgboost', xgboost_clf),
                ('adaboost', adaboost_clf)
            ],
            final_estimator=meta_learner,  # Meta-learner
            cv=5,                          # Cross-validation folds for training meta-learner
            passthrough=False,             # Don't pass original features to meta-learner
            n_jobs=-1                      # Use all available processors
        )

        # Define parameter grid for meta-learner tuning
        param_grid = {
            'classifier__final_estimator__C': [0.1, 1, 10]  # Regularization parameter for logistic regression
        }

        return stacked_clf, param_grid

    def preprocessing_pipeline(self, numerical_cols, categorical_cols):
        """
        Builds the complete preprocessing pipeline for the dataset.
        
        The pipeline includes:
        1. Feature engineering (splitting columns, creating new features)
        2. Logical imputation (domain-specific missing value filling)
        3. Classical preprocessing (scaling, encoding, remaining imputation)
        4. Boolean to integer conversion
        
        Args:
            numerical_cols (list): List of numerical column names
            categorical_cols (list): List of categorical column names
            
        Returns:
            Pipeline: Complete preprocessing pipeline
        """
        # === USER INPUT FOR PREPROCESSING CHOICES ===
        print("\n=== PREPROCESSING CONFIGURATION ===")
        
        # Get user choice for scaling method
        scaler_choice = input("Choose scaling method:\n1. StandardScaler\n2. MinMaxScaler\n3. No scaling\nEnter choice (1/2/3): ").strip()
        scaler = MinMaxScaler() if scaler_choice == '2' else StandardScaler() if scaler_choice == '1' else None

        # Get user choice for numerical imputation strategy
        imputation_numerical_choice = input("Choose imputation strategy:\n1. Mean\n2. KNN\nEnter choice (1/2): ").strip()
        imputation_numerical = SimpleImputer(strategy='mean') if imputation_numerical_choice == '1' else KNNImputer(n_neighbors=5)

        # === NUMERICAL FEATURES PIPELINE ===
        # Handle numerical features: imputation → scaling
        numerical_transformer = Pipeline(steps=[
            ('imputer', imputation_numerical),  # Fill missing values
            ('scaler', scaler) if scaler else ('passthrough', 'passthrough')  # Scale features (optional)
        ])

        # === CATEGORICAL FEATURES PIPELINE ===
        # Handle categorical features: imputation → one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with mode
            ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Convert to binary features
        ])

        # === MAIN PREPROCESSOR ===
        # Apply different transformations to different column types
        main_preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),   # Apply numerical pipeline to numerical columns
            ('cat', categorical_transformer, categorical_cols) # Apply categorical pipeline to categorical columns
        ])

        # === COMPLETE PREPROCESSING PIPELINE ===
        # Chain all preprocessing steps in logical order
        full_preprocessor = Pipeline(steps=[
            ('feature_engineering', FeatureEngineeringTransformer()),  # Create new features
            ('logical_imputation', LogicalImputationTransformer()),    # Apply domain-specific imputation
            ('preprocessing', main_preprocessor),                      # Classical preprocessing
            ('bool2int', TransformBooleanToInt())                     # Convert booleans to integers
        ])

        return full_preprocessor

    def train_and_evaluate_model(self, choice, X_train, y_train, X_val, y_val):
        """
        Complete model training and evaluation workflow.
        
        This method:
        1. Builds the preprocessing pipeline
        2. Creates the selected model
        3. Performs hyperparameter tuning with cross-validation
        4. Evaluates the model on training and validation sets
        5. Generates visualizations and performance metrics
        
        Args:
            choice (str): Model type choice ('1'-'6')
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            sklearn.Pipeline: Best trained model pipeline
        """
        print("\n=== MODEL TRAINING AND EVALUATION ===")
        
        # === DEFINE COLUMN CATEGORIES ===
        # Note: These lists include features that will be created by feature engineering
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'TotalSpent', 'FamilySize']
        categorical_cols = ['Deck', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']

        # === BUILD PREPROCESSING PIPELINE ===
        preprocessor = self.preprocessing_pipeline(numerical_cols, categorical_cols)
        
        # === CREATE MODEL AND PARAMETER GRID ===
        model, param_grid = self.create_model(choice)

        # === CREATE COMPLETE ML PIPELINE ===
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Data preprocessing
            ('classifier', model)            # Machine learning model
        ])

        # === SETUP CROSS-VALIDATION ===
        # Use stratified k-fold to maintain class balance in each fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # === HYPERPARAMETER TUNING ===
        print("Starting hyperparameter tuning...")
        
        # Use RandomizedSearchCV for CatBoost (too many parameters), GridSearchCV for others
        if choice == '5':
            # CatBoost: Use randomized search due to large parameter space
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=50,              # Number of parameter combinations to try
                cv=cv,                  # Cross-validation strategy
                scoring='accuracy',     # Optimization metric
                n_jobs=-1,             # Use all available processors
                verbose=2,             # Print progress
                random_state=42        # For reproducible results
            )
        else:
            # Other models: Use grid search for exhaustive parameter exploration
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,                 # Cross-validation strategy
                scoring='accuracy',    # Optimization metric
                n_jobs=-1             # Use all available processors
            )

        # === MODEL TRAINING ===
        print("Training model with cross-validation...")
        search.fit(X_train, y_train)

        # === DISPLAY BEST PARAMETERS ===
        print("\n=== HYPERPARAMETER TUNING RESULTS ===")
        print("Best parameters:", search.best_params_)
        print("Best cross-validated accuracy:", search.best_score_)

        # === MODEL EVALUATION ===
        print("\n=== MODEL EVALUATION ===")
        
        # Make predictions on both training and validation sets
        y_pred = search.best_estimator_.predict(X_val)           # Validation predictions
        y_pred_train = search.best_estimator_.predict(X_train)   # Training predictions

        # Calculate and display accuracy scores
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Accuracy on training set: {train_accuracy:.4f}")
        print(f"Accuracy on validation set: {val_accuracy:.4f}")
        print("Classification report:\n", classification_report(y_val, y_pred))

        # === VISUALIZATION ===
        print("\n=== GENERATING VISUALIZATIONS ===")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # === CONFUSION MATRIX ===
        conf_matrix = confusion_matrix(y_val, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix on Validation Set")

        # === ROC CURVE ===
        # Only generate ROC curve if model supports probability prediction
        if hasattr(search.best_estimator_.named_steps['classifier'], "predict_proba"):
            y_proba = search.predict_proba(X_val)[:, 1]  # Get probabilities for positive class
            fpr, tpr, _ = roc_curve(y_val, y_proba)      # Calculate ROC curve points
            roc_auc = auc(fpr, tpr)                      # Calculate area under curve

            # Plot ROC curve
            ax2.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
            ax2.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal reference line
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Receiver Operating Characteristic (ROC)')
            ax2.legend(loc='lower right')
        else:
            # Display message if ROC curve is not available
            ax2.text(0.5, 0.5, 'ROC curve not available\nfor this model',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ROC Curve - Not Available')

        plt.tight_layout()
        plt.show()

        # === FEATURE IMPORTANCE ANALYSIS ===
        # Display feature importances for tree-based models
        if hasattr(search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
            print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
            self.feature_importances(search.best_estimator_, X_train)

        return search.best_estimator_

    def feature_importances(self, model, X_train):
        """
        Compute and visualize feature importances for tree-based models.
        
        This method:
        1. Extracts feature importances from the trained model
        2. Maps importances to meaningful feature names
        3. Groups related features (e.g., one-hot encoded categories)
        4. Creates a visualization of feature importance rankings
        
        Args:
            model: Trained pipeline with feature importance capability
            X_train: Training data (used for feature name extraction)
        """
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            # Extract feature importances from the model
            importances = model.named_steps['classifier'].feature_importances_

            try:
                # === EXTRACT FEATURE NAMES ===
                main_preprocessor = model.named_steps['preprocessor'].named_steps['preprocessing']
                feature_names = []

                # Add numerical feature names
                numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'TotalSpent', 'FamilySize']
                feature_names.extend(numerical_cols)

                # Add categorical feature names (after one-hot encoding)
                categorical_cols = ['Deck', 'Side', 'CryoSleep', 'HomePlanet', 'Destination', 'VIP']
                # Try different methods to get feature names (sklearn version compatibility)
                if hasattr(main_preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
                    cat_feature_names = main_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
                else:
                    cat_feature_names = main_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_cols)

                feature_names.extend(cat_feature_names)

            except Exception as e:
                # Fallback to generic feature names if extraction fails
                print(f"Could not extract feature names: {e}")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            # === VALIDATE FEATURE NAMES ===
            if len(feature_names) != len(importances):
                print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of importances ({len(importances)})")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            # === GROUP RELATED FEATURES ===
            # Group one-hot encoded features by their base category
            grouped_importances = {}
            for i, feature_name in enumerate(feature_names):
                # Check if this is a one-hot encoded feature (contains underscore but not engineered features)
                if '_' in feature_name and feature_name not in ['Cabin_num', 'TotalSpent', 'FamilySize']:
                    # Extract base category name (e.g., "HomePlanet_Earth" → "HomePlanet")
                    base_name = '_'.join(feature_name.split('_')[:-1])
                    grouped_importances[base_name] = grouped_importances.get(base_name, 0) + importances[i]
                else:
                    # Keep original feature name for numerical and engineered features
                    grouped_importances[feature_name] = importances[i]

            # === PREPARE DATA FOR VISUALIZATION ===
            grouped_names = list(grouped_importances.keys())
            grouped_values = list(grouped_importances.values())
            # Sort features by importance (descending order)
            indices = sorted(range(len(grouped_values)), key=lambda i: grouped_values[i], reverse=True)

            # === CREATE FEATURE IMPORTANCE PLOT ===
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
