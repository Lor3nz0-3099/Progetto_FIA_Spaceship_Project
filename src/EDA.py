import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Pass `\\(name,\\)` instead of `name`.*", category=FutureWarning)

# Global style
sns.set_style('whitegrid')


def plot_age_distribution(df, save=False, save_dir='plots'):
    """
    It shows complete distribution of the age feature.
    """

    plt.figure(figsize=(6, 5))
    sns.histplot(x=df['Age'], kde=True, color=(0.2, 0.6, 0.8), linewidth=0)
    plt.title('Age Distribution', fontsize=12)
    plt.xlabel('Age', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/eta_completa.png")
    plt.show()

def plot_univariate_analysis(df, categorical_cols, target_col, save=False, save_dir='plots'):
    """
    It creates a bar chart and a pie chart for each categorical variable,
    including the target variable, to show the distribution of values.
    """

    
    for col in categorical_cols + [target_col]:
        plt.figure(figsize=(12, 4))

        # Bar chart: shows the absolute frequency of values
        plt.subplot(1, 2, 1)
        sns.countplot(x=df[col], palette='Set2')
        plt.title(f'Distribuzione di {col}', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Frequency', fontsize=10)

        # Pie chart: shows the percentage distribution of values
        plt.subplot(1, 2, 2)
        df[col].value_counts().plot(
            kind='pie',
            autopct='%.2f%%',
            shadow=True,
            colors=sns.color_palette('Set2')
        )
        plt.title(f'{col} percentage distribution', fontsize=12)
        plt.ylabel('')
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{col}_univariate.png")
        plt.show()


def plot_numerical_distribution(df, numeric_cols, save=False, save_dir='plots'):
    for col in numeric_cols:
        if col == 'Age':
            continue
        plt.figure(figsize=(6, 5))

        # Calcolo del limite massimo al 80° percentile
        upper_limit = df[col].quantile(0.80)

        # Filtro per mostrare solo valori "normali"
        filtered = df[df[col] <= upper_limit]

        sns.histplot(x=filtered[col], kde=True, color=(1.0, 0.4, 0.4), linewidth=0)
        plt.title(f'{col} distribution (to the 80° percentile)', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{col}_numerical_distribution.png")
        plt.show()

def plot_target_vs_categorical(df, categorical_cols, target_col, save=False, save_dir='plots'):
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue=target_col, palette='Set1')
        plt.title(f'{col} as a function of {target_col}', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{col}_vs_{target_col}.png")
        plt.show()

        # Calcolo e stampa delle percentuali per ogni combinazione
        percentuali = df.groupby([col, target_col])[target_col].count() / df.groupby([col])[target_col].count()
        print(f"\n{col} conditional percentage distribution:\n", percentuali.mul(100).round(2))

def plot_target_vs_numerical(df, numeric_cols, target_col, save=False, save_dir='plots'):
    """
    For each numeric column, it shows a histogram with the distribution of values
    and a boxplot to visualize the distribution of values for each class of the target variable.
    """
    for col in numeric_cols:
        plt.figure(figsize=(10, 4.8))
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x=col, hue=target_col, kde=True, palette='Set1', stat='density', binwidth=8)
        plt.title(f'{col} distribution for class {target_col}', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel("Density", fontsize=10)

        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, y=col, x=target_col, palette='Set1')
        plt.title(f'{col} for {target_col} (Boxplot)', fontsize=12)
        plt.xlabel(target_col, fontsize=10)
        plt.ylabel(col, fontsize=10)
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{col}_vs_{target_col}_boxplot.png")
        plt.show()


def plot_average_expenses_by_target(df, expenses_cols, target_col, save=False, save_dir='plots'):
    """
    It shows the average expenses for each category
    based on the target variable (Transported).
    """

    medie_spesa = df.groupby(target_col)[expenses_cols].mean()
    plt.figure(figsize=(10, 5))
    medie_spesa.plot(kind='bar')
    plt.title('Average expense for category', fontsize=12)
    plt.ylabel('Average expense', fontsize=10)
    plt.xlabel('Target Class (Transported)', fontsize=10)
    plt.xticks(rotation=0)
    plt.legend(expenses_cols, title="Tipo di spesa")
    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/average_expenses_by_{target_col}.png")
    plt.show()

# ============================================
# RELAZIONE TRA VARIABILI CATEGORICHE
# ============================================

def plot_categorical_vs_categorical(df, categorical_cols, save=False, save_dir='plots'):
    """
    For each categorical column, it creates a count plot comparing it with all other categorical columns.
    This allows to visualize the relationship between different categorical variables.
    """

    for col1 in categorical_cols:
        plt.figure(figsize=(15, 4))
        i = 1
        for col2 in categorical_cols:
            if col1 == col2:
                continue
            plt.subplot(1, len(categorical_cols) - 1, i)
            sns.countplot(data=df, x=col1, hue=col2)
            plt.title(f'{col1} relative to {col2}', fontsize=10)
            plt.xlabel(col1)
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            i += 1
        plt.suptitle(f'{col1} compared with other categorical values', fontsize=12)
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{col1}_vs_others.png")
        plt.show()


def run_eda(train_df, save=True, save_dir='plots'):
    """
    Principal function to run the exploratory data analysis (EDA) on the training dataset.
    It generates various plots to visualize the data distribution, relationships between features,
    """
    # Definizione delle colonne categoriche e numeriche nei dati originali

    categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    numerical = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    target = 'Transported'

    print(" Starting EDA...\n")

    plot_univariate_analysis(train_df, categorical, target, save=save, save_dir=save_dir)
    plot_age_distribution(train_df, save=save, save_dir=save_dir)
    plot_numerical_distribution(train_df, numerical, save=save, save_dir=save_dir)
    plot_target_vs_categorical(train_df, categorical, target, save=save, save_dir=save_dir)
    plot_target_vs_numerical(train_df, ['Age'], target, save=save, save_dir=save_dir)
    plot_average_expenses_by_target(train_df, numerical, target, save=save, save_dir=save_dir)
    plot_categorical_vs_categorical(train_df, categorical, save=save, save_dir=save_dir)

    print("\n EDA completed, Roger Roger!\n")

