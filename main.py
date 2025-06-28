import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
dataset_df = pd.read_csv('C:\\Users\\loren\\Desktop\\Progetto_FIA_Spaceship\\train.csv')
print("Full dataset shape: {}".format(dataset_df.shape))
print(dataset_df.head(5))

# Bar chart for label distribution
plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind="bar")
plt.title("Label distribution")
plt.show()

# Drop unnecessary columns and split features
dataset_df = dataset_df.drop(['PassengerId'], axis=1)
dataset_df[['Deck', 'Cabin_num', 'Side']] = dataset_df['Cabin'].str.split('/', expand=True)
dataset_df[['Name_', 'Surname']] = dataset_df['Name'].str.split(' ', expand=True)
dataset_df = dataset_df.drop(['Name', 'Cabin', 'Name_'], axis=1)
print(dataset_df.head(5))

# Count missing values and print them for each column
missing_values = dataset_df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Fill miossing values of cabin
for i in range(len(dataset_df)):
    if pd.isnull(dataset_df.loc[i, 'Surname']):
        cabin_num = dataset_df.loc[i, 'Cabin_num']
        dataset_df.loc[i, 'Surname'] = dataset_df.loc[dataset_df['Cabin_num'] == cabin_num, 'Surname'].mode()[0] if not dataset_df.loc[dataset_df['Cabin_num'] == cabin_num, 'Surname'].mode().empty else np.nan
    elif pd.isnull(dataset_df.loc[i, 'Deck']):
        surname = dataset_df.loc[i, 'Surname']
        dataset_df.loc[i, 'Deck'] = dataset_df.loc[dataset_df['Surname'] == surname, 'Deck'].mode()[0] if not dataset_df.loc[dataset_df['Surname'] == surname, 'Deck'].mode().empty else np.nan
        dataset_df.loc[i, 'Cabin_num'] = dataset_df.loc[dataset_df['Surname'] == surname, 'Cabin_num'].mode()[0] if not dataset_df.loc[dataset_df['Surname'] == surname, 'Cabin_num'].mode().empty else np.nan
        dataset_df.loc[i, 'Side'] = dataset_df.loc[dataset_df['Surname'] == surname, 'Side'].mode()[0] if not dataset_df.loc[dataset_df['Surname'] == surname, 'Side'].mode().empty else np.nan
    elif pd.isnull(dataset_df.loc[i, 'HomePlanet']):
        surname = dataset_df.loc[i, 'Surname']
        dataset_df.loc[i, 'HomePlanet'] = dataset_df.loc[dataset_df['Surname'] == surname, 'HomePlanet'].mode()[0] if not dataset_df.loc[dataset_df['Surname'] == surname, 'HomePlanet'].mode().empty else np.nan
    elif pd.isnull(dataset_df.loc[i, 'Destination']):
        surname = dataset_df.loc[i, 'Surname']
        dataset_df.loc[i, 'Destination'] = dataset_df.loc[dataset_df['Surname'] == surname, 'Destination'].mode()[0] if not dataset_df.loc[dataset_df['Surname'] == surname, 'Destination'].mode().empty else np.nan
    elif pd.isnull(dataset_df.loc[i, 'VIP']):
        surname = dataset_df.loc[i, 'Surname']
        dataset_df.loc[i, 'VIP'] = dataset_df.loc[dataset_df['Surname'] == surname, 'VIP'].mode()[0] if not dataset_df.loc[dataset_df['Surname'] == surname, 'VIP'].mode().empty else np.nan
    elif pd.isnull(dataset_df.loc[i, 'CryoSleep']):
        expense = dataset_df.loc[i, 'RoomService'] + dataset_df.loc[i, 'FoodCourt'] + dataset_df.loc[i, 'ShoppingMall'] + dataset_df.loc[i, 'Spa'] + dataset_df.loc[i, 'VRDeck']
        if expense > 0:
            dataset_df.loc[i, 'CryoSleep'] = False
        else:
            dataset_df.loc[i, 'CryoSleep'] = True
          
missing_values = dataset_df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Drop rows with Nan values in 'Deck', 'Cabin_num', and 'Side'
dataset_df = dataset_df.dropna(subset=['Deck', 'Cabin_num', 'Side', 'HomePlanet', 'Destination', 'VIP', 'CryoSleep'])
dataset_df = dataset_df.drop(['Surname'], axis=1)
missing_values = dataset_df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Fill missing values for numerical columns with their mean
numerical_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
for col in numerical_cols:
    dataset_df[col] = dataset_df[col].fillna(value=dataset_df[col].mean())

# Convert boolean to integer
dataset_df['Transported'] = dataset_df['Transported'].astype(int)
dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)

missing_values = dataset_df.isnull().sum()
print("Missing values in each column after imputation:")
print(missing_values[missing_values > 0])

# Split features and label
label = 'Transported'
X = dataset_df.drop(label, axis=1)
y = dataset_df[label]

# One-hot encoding
X = pd.get_dummies(X)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) #random_state=42 ensures reproducibility which means the same split will be generated every time the code is run. 

# Align columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Grid search with Stratified K-Fold
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5] 
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validated accuracy:", grid_search.best_score_)

# Final evaluation on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test set accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

importances = best_model.feature_importances_
features = X_train.columns
imp_df = pd.Series(importances, index=features).sort_values(ascending=False)[:20]
imp_df.plot(kind='barh')
plt.title("Top 20 Feature Importances")
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()






