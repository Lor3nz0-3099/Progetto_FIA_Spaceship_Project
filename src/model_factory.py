from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
        Includes imputers and scalers based on user input.
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

        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        return preprocessor

    def train_and_evaluate_model(self, choice, X_train, y_train, X_val, y_val):
        """
        Builds the full pipeline, runs GridSearchCV, fits on training data,
        and evaluates on training and validation sets.
        """
        # Feature lists (customize as needed)
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

        # Confusion matrix plot
        conf_matrix = confusion_matrix(y_val, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix on Validation Set")
        plt.show()

        return grid_search.best_estimator_
