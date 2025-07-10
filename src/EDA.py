import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Pass `\\(name,\\)` instead of `name`.*", category=FutureWarning)

# Global style
sns.set_style('whitegrid')


class EDAAnalyzer:
    """
    A class for performing Exploratory Data Analysis on the Spaceship Titanic dataset.
    """
    
    def __init__(self, save_dir='plots'):
        """
        Initialize the EDA analyzer.
        
        Args:
            save_dir (str): Directory to save plots
        """
        self.save_dir = save_dir
        self.categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
        self.numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.target_col = 'Transported'
        
    def plot_age_distribution(self, df, save=False):
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
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(f"{self.save_dir}/eta_completa.png")
        plt.show()

    def plot_univariate_analysis(self, df, categorical_cols, target_col, save=False):
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
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_univariate.png")
            plt.show()


    def plot_numerical_distribution(self, df, numeric_cols, save=False):
        for col in numeric_cols:
            if col == 'Age':
                continue
            plt.figure(figsize=(6, 5))

            # Calculate the upper limit for the 80th percentile
            # to filter out extreme values
            upper_limit = df[col].quantile(0.80)

            # Filter to show only "normal" values
            # This helps to visualize the distribution without outliers
            filtered = df[df[col] <= upper_limit]

            sns.histplot(x=filtered[col], kde=True, color=(1.0, 0.4, 0.4), linewidth=0)
            plt.title(f'{col} distribution (to the 80° percentile)', fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.tight_layout()
            if save:
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_numerical_distribution.png")
            plt.show()

    def plot_target_vs_categorical(self, df, categorical_cols, target_col, save=False):
        for col in categorical_cols:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x=col, hue=target_col, palette='Set1')
            plt.title(f'{col} as a function of {target_col}', fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save:
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_vs_{target_col}.png")
            plt.show()

            # Calculate the percentage distribution of the target variable for each category
            # This shows how the target variable is distributed within each category
            percentuali = df.groupby([col, target_col])[target_col].count() / df.groupby([col])[target_col].count()
            print(f"\n{col} conditional percentage distribution:\n", percentuali.mul(100).round(2))

    def plot_target_vs_numerical(self, df, numeric_cols, target_col, save=False):
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
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_vs_{target_col}_boxplot.png")
            plt.show()


    def plot_average_expenses_by_target(self, df, expenses_cols, target_col, save=False):
        """
        It shows the average expenses for each category
        based on the target variable (Transported).
        """
        medie_spesa = df.groupby(target_col)[expenses_cols].mean()
        
        # Create the plot directly without creating an empty figure first
        ax = medie_spesa.plot(kind='bar', figsize=(10, 5))
        ax.set_title('Average expense for category', fontsize=12)
        ax.set_ylabel('Average expense', fontsize=10)
        ax.set_xlabel('Target Class (Transported)', fontsize=10)
        ax.tick_params(axis='x', rotation=0)
        ax.legend(expenses_cols, title="Tipo di spesa")
        plt.tight_layout()
        if save:
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(f"{self.save_dir}/average_expenses_by_{target_col}.png")
        plt.show()


    def run_eda(self, train_df, save=True):
        """
        Principal function to run the exploratory data analysis (EDA) on the training dataset.
        It generates various plots to visualize the data distribution, relationships between features,
        """
        print(" Starting EDA...\n")

        self.plot_univariate_analysis(train_df, self.categorical_cols, self.target_col, save=save)
        self.plot_age_distribution(train_df, save=save)
        self.plot_numerical_distribution(train_df, self.numerical_cols, save=save)
        self.plot_target_vs_categorical(train_df, self.categorical_cols, self.target_col, save=save)
        self.plot_target_vs_numerical(train_df, ['Age'], self.target_col, save=save)
        
        # Only expense-related columns (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck) are selected for average expenses
        self.plot_average_expenses_by_target(train_df, self.numerical_cols[1:], self.target_col, save=save) 

        print("\n EDA completed, Roger Roger!\n")

