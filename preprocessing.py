import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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







