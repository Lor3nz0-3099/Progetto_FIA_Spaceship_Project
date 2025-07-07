import pandas as pd
import numpy as np

class XGBPreprocessor: # Preprocessing class for XGBoost
    def __init__(self, scaler=None):
        self.scaler = scaler
        self.train_columns = None
        self.num_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
        self.cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']

    def _feature_engineering(self, df, is_train=True): # Feature engineering method
        df[['Deck', 'Cabin_num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df['Cabin_num'] = pd.to_numeric(df['Cabin_num'], errors='coerce').fillna(0)
        df[['Name_', 'Surname']] = df['Name'].str.split(' ', expand=True)
        df.drop(['PassengerId', 'Cabin', 'Name', 'Name_'], axis=1, inplace=True)

        if is_train:
            # clever handling of missing values based on Surname
            for i in range(len(df)):
                if pd.isnull(df.loc[i, 'Surname']):
                    cabin_num = df.loc[i, 'Cabin_num']
                    mode = df[df['Cabin_num'] == cabin_num]['Surname'].mode()
                    df.loc[i, 'Surname'] = mode[0] if not mode.empty else None

                surname = df.loc[i, 'Surname']
                if pd.isnull(df.loc[i, 'Deck']):
                    for col in ['Deck', 'Cabin_num', 'Side']:
                        mode = df[df['Surname'] == surname][col].mode()
                        if not mode.empty:
                            df.loc[i, col] = mode[0]
                if pd.isnull(df.loc[i, 'HomePlanet']):
                    mode = df[df['Surname'] == surname]['HomePlanet'].mode()
                    if not mode.empty:
                        df.loc[i, 'HomePlanet'] = mode[0]
                if pd.isnull(df.loc[i, 'Destination']):
                    mode = df[df['Surname'] == surname]['Destination'].mode()
                    if not mode.empty:
                        df.loc[i, 'Destination'] = mode[0]
                if pd.isnull(df.loc[i, 'VIP']):
                    mode = df[df['Surname'] == surname]['VIP'].mode()
                    if not mode.empty:
                        df.loc[i, 'VIP'] = mode[0]
                if pd.isnull(df.loc[i, 'CryoSleep']):
                    total_exp = df.loc[i, self.num_cols].sum()
                    df.loc[i, 'CryoSleep'] = total_exp == 0

            # Fallback if still NaN after mode filling
            df['VIP'].fillna(False, inplace=True)
            df['CryoSleep'].fillna(False, inplace=True)

        else:
            # for test set, fill NaN values with defaults
            df['VIP'].fillna(False, inplace=True)
            df['CryoSleep'].fillna(False, inplace=True)
            df['HomePlanet'].fillna('Unknown', inplace=True)
            df['Destination'].fillna('Unknown', inplace=True)
            df['Deck'].fillna('Unknown', inplace=True)
            df['Side'].fillna('Unknown', inplace=True)

        # Cast booleani
        df['VIP'] = df['VIP'].astype(int)
        df['CryoSleep'] = df['CryoSleep'].astype(int)

        # fill missing numerical values with mean
        df[self.num_cols] = df[self.num_cols].fillna(df[self.num_cols].mean())

        # remove 'Surname' column
        df.drop(['Surname'], axis=1, inplace=True)

        # One-hot encoding
        df = pd.get_dummies(df, columns=self.cat_cols, drop_first=True)

        return df

    def fit(self, X_path, y=None):
        df = pd.read_csv(X_path)
        df = self._feature_engineering(df, is_train=True)
        self.train_columns = [col for col in df.columns if col != 'Transported']

        if self.scaler is not None:
            self.scaler.fit(df[self.num_cols])

        return self

    def transform(self, X_path):
        df = pd.read_csv(X_path)
        df = self._feature_engineering(df, is_train=False)

        # Applica lo scaler
        if self.scaler is not None:
            df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # Aggiungi colonne mancanti
        for col in self.train_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.train_columns]

        return df

    def fit_transform(self, X_path, y=None):
        df = pd.read_csv(X_path)
        df = self._feature_engineering(df, is_train=True)

        self.train_columns = [col for col in df.columns if col != 'Transported']

        if self.scaler is not None:
            self.scaler.fit(df[self.num_cols])
            df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        X = df.drop('Transported', axis=1)
        y = df['Transported'].astype(int)

        return X, y