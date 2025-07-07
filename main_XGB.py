import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from preprocessingXGBoost import XGBPreprocessor
# 1. Preprocessing
preprocessor = XGBPreprocessor()
X, y = preprocessor.fit_transform('/Users/claudia/Desktop/cartella senza nome 3/train.csv')

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Grid Search per XGBoost (opzionale: puoi attivarlo o no)
use_grid_search = True

if use_grid_search:
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters from GridSearchCV:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
else:
    best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    best_model.fit(X_train, y_train)

# 4. prediction and evaluation
y_pred_xgb = best_model.predict(X_test)
print("XGBoost Test set accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification report:\n", classification_report(y_test, y_pred_xgb))

# 5. Feature importance (gain-based)
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feat_imp[:20].plot(kind='barh', color='purple')
plt.title("Top Feature Importances - XGBoost")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()

# 6. confusion matrix
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Purples')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")
plt.show()