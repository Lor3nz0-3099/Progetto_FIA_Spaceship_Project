import pandas as pd                    
import seaborn as sns                  
import matplotlib.pyplot as plt       
import os                             

# Suppress future warnings to keep output clean
import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Pass `\\(name,\\)` instead of `name`.*", category=FutureWarning)

# Set global plotting style for consistent appearance
sns.set_style('whitegrid')


class EDAAnalyzer:
    """
    A class for performing Exploratory Data Analysis on the Spaceship Titanic dataset.
    
    This class provides comprehensive visualization and analysis tools to understand:
    - Data distributions and patterns
    - Relationships between features and target variable
    - Feature importance insights
    """
    
    def __init__(self, save_dir='plots'):
        """
        Initialize the EDA analyzer with predefined column categories.
        
        Args:
            save_dir (str): Directory to save generated plots
        """
        self.save_dir = save_dir
        
        # Define categorical features for analysis
        self.categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
        
        # Define numerical features for analysis
        self.numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Define target variable
        self.target_col = 'Transported'
        
    def plot_age_distribution(self, df, save=False):
        """
        Visualize the complete distribution of the Age feature.
        
        Creates a histogram with kernel density estimation to show:
        - Overall age distribution patterns
        - Most common age ranges
        - Data skewness and outliers
        
        Args:
            df: Input DataFrame containing the data
            save: Whether to save the plot to file
        """
        # Create figure with specified size
        plt.figure(figsize=(6, 5))
        
        # Create histogram with kernel density estimation overlay
        sns.histplot(x=df['Age'], kde=True, color=(0.2, 0.6, 0.8), linewidth=0)
        
        # Add plot labels and title
        plt.title('Age Distribution', fontsize=12)
        plt.xlabel('Age', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(f"{self.save_dir}/age_distribution.png")
        plt.show()

    def plot_univariate_analysis(self, df, categorical_cols, target_col, save=False):
        """
        Create comprehensive univariate analysis for categorical variables.
        
        For each categorical variable (including target), generates:
        - Bar chart: Shows absolute frequency of each category
        - Pie chart: Shows percentage distribution of categories
        
        This helps understand:
        - Category balance/imbalance
        - Most and least common categories
        - Overall data composition
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            target_col: Target variable column name
            save: Whether to save plots to file
        """
        # Analyze each categorical column plus the target variable
        for col in categorical_cols + [target_col]:
            plt.figure(figsize=(12, 4))

            # === LEFT SUBPLOT: BAR CHART ===
            plt.subplot(1, 2, 1)
            # Create count plot showing absolute frequencies
            sns.countplot(x=df[col], palette='Set2')
            plt.title(f'{col} distribution', fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel('Frequency', fontsize=10)

            # === RIGHT SUBPLOT: PIE CHART ===
            plt.subplot(1, 2, 2)
            # Create pie chart showing percentage distribution
            df[col].value_counts().plot(
                kind='pie',
                autopct='%.2f%%',    # Show percentages with 2 decimal places
                shadow=True,         # Add shadow for visual appeal
                colors=sns.color_palette('Set2')
            )
            plt.title(f'{col} percentage distribution', fontsize=12)
            plt.ylabel('')  # Remove default ylabel for pie charts
            plt.tight_layout()
            
            # Save plot if requested
            if save:
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_univariate.png")
            plt.show()

    def plot_numerical_distribution(self, df, numeric_cols, save=False):
        """
        Visualize distributions of numerical features (excluding Age).
        
        For each numerical column, creates a histogram with KDE to show:
        - Distribution shape and skewness
        - Central tendency and spread
        - Presence of outliers
        
        Uses 80th percentile filtering to focus on typical values
        and avoid extreme outliers that could skew visualization.
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numerical column names
            save: Whether to save plots to file
        """
        for col in numeric_cols:
            # Skip Age as it's handled separately
            if col == 'Age':
                continue
                
            plt.figure(figsize=(6, 5))

            # Calculate the 80th percentile to filter out extreme outliers
            # This helps visualize the main distribution without being skewed by outliers
            upper_limit = df[col].quantile(0.80)

            # Filter data to show only "typical" values (up to 80th percentile)
            # This provides a clearer view of the main distribution pattern
            filtered = df[df[col] <= upper_limit]

            # Create histogram with kernel density estimation
            sns.histplot(x=filtered[col], kde=True, color=(1.0, 0.4, 0.4), linewidth=0)
            plt.title(f'{col} distribution (to the 80° percentile)', fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.tight_layout()
            
            # Save plot if requested
            if save:
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_numerical_distribution.png")
            plt.show()

    def plot_target_vs_categorical(self, df, categorical_cols, target_col, save=False):
        """
        Analyze relationships between categorical features and target variable.
        
        For each categorical feature, creates:
        - Grouped bar chart showing target distribution within each category
        - Percentage analysis showing conditional probabilities
        
        This reveals:
        - Which categories are more/less likely to be transported
        - Feature importance for prediction
        - Class imbalances within categories
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            target_col: Target variable column name
            save: Whether to save plots to file
        """
        for col in categorical_cols:
            plt.figure(figsize=(6, 4))
            
            # Create grouped bar chart with target variable as hue
            sns.countplot(data=df, x=col, hue=target_col, palette='Set1')
            plt.title(f'{col} as a function of {target_col}', fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.xticks(rotation=45)  # Rotate x-axis labels for readability
            plt.tight_layout()
            
            # Save plot if requested
            if save:
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_vs_{target_col}.png")
            plt.show()

            # === CONDITIONAL PROBABILITY ANALYSIS ===
            # Calculate the percentage distribution of target variable for each category
            # This shows the probability of being transported given each category value
            percentuali = df.groupby([col, target_col])[target_col].count() / df.groupby([col])[target_col].count()
            print(f"\n{col} conditional percentage distribution:\n", percentuali.mul(100).round(2))

    def plot_target_vs_numerical(self, df, numeric_cols, target_col, save=False):
        """
        Analyze relationships between numerical features and target variable.
        
        For each numerical column, creates:
        - Histogram with target-based color coding (shows distribution overlap)
        - Boxplot showing distribution differences between target classes
        
        This reveals:
        - Whether numerical features have different distributions for each target class
        - Presence of outliers in different classes
        - Potential thresholds for classification
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numerical column names
            target_col: Target variable column name
            save: Whether to save plots to file
        """
        for col in numeric_cols:
            plt.figure(figsize=(10, 4.8))
            
            # === LEFT SUBPLOT: OVERLAPPING HISTOGRAMS ===
            plt.subplot(1, 2, 1)
            # Create density histograms separated by target class
            sns.histplot(data=df, x=col, hue=target_col, kde=True, palette='Set1', 
                        stat='density', binwidth=8)
            plt.title(f'{col} distribution for class {target_col}', fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Density", fontsize=10)

            # === RIGHT SUBPLOT: BOXPLOT COMPARISON ===
            plt.subplot(1, 2, 2)
            # Create boxplot to compare distributions between target classes
            sns.boxplot(data=df, y=col, x=target_col, palette='Set1')
            plt.title(f'{col} for {target_col} (Boxplot)', fontsize=12)
            plt.xlabel(target_col, fontsize=10)
            plt.ylabel(col, fontsize=10)
            plt.tight_layout()
            
            # Save plot if requested
            if save:
                os.makedirs(self.save_dir, exist_ok=True)
                plt.savefig(f"{self.save_dir}/{col}_vs_{target_col}_boxplot.png")
            plt.show()

    def plot_average_expenses_by_target(self, df, expenses_cols, target_col, save=False):
        """
        Analyze spending patterns between transported and non-transported passengers.
        
        Creates a bar chart showing average expenses in each category
        (RoomService, FoodCourt, etc.) grouped by target variable.
        
        This reveals:
        - Whether transported passengers spend more/less on services
        - Which services show the biggest differences between groups
        - Overall spending behavior patterns
        
        Args:
            df: Input DataFrame
            expenses_cols: List of expense-related column names
            target_col: Target variable column name
            save: Whether to save plot to file
        """
        # Calculate average expenses for each target class
        medie_spesa = df.groupby(target_col)[expenses_cols].mean()
        
        # Create grouped bar chart showing average expenses
        ax = medie_spesa.plot(kind='bar', figsize=(10, 5))
        ax.set_title('Average expense for category', fontsize=12)
        ax.set_ylabel('Average expense', fontsize=10)
        ax.set_xlabel('Target Class (Transported)', fontsize=10)
        ax.tick_params(axis='x', rotation=0)  # Keep x-axis labels horizontal
        ax.legend(expenses_cols, title="Tipo di spesa")  # Add legend for expense types
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(f"{self.save_dir}/average_expenses_by_{target_col}.png")
        plt.show()

    def run_eda(self, train_df, save=True):
        """
        Execute complete exploratory data analysis workflow.
        
        Runs all EDA methods in logical sequence to provide comprehensive
        data understanding:
        1. Categorical variable analysis (univariate)
        2. Age distribution analysis
        3. Numerical variable distributions
        4. Categorical vs target relationships
        5. Numerical vs target relationships  
        6. Expense pattern analysis
        
        Args:
            train_df: Training DataFrame to analyze
            save: Whether to save all generated plots to files
        """
        print("=== Starting EDA Analysis ===\n")

        # === UNIVARIATE ANALYSIS ===
        print("1. Analyzing categorical variable distributions...")
        self.plot_univariate_analysis(train_df, self.categorical_cols, self.target_col, save=save)
        
        # === AGE ANALYSIS ===
        print("2. Analyzing age distribution...")
        self.plot_age_distribution(train_df, save=save)
        
        # === NUMERICAL DISTRIBUTIONS ===
        print("3. Analyzing numerical variable distributions...")
        self.plot_numerical_distribution(train_df, self.numerical_cols, save=save)
        
        # === BIVARIATE ANALYSIS: CATEGORICAL VS TARGET ===
        print("4. Analyzing categorical features vs target...")
        self.plot_target_vs_categorical(train_df, self.categorical_cols, self.target_col, save=save)
        
        # === BIVARIATE ANALYSIS: AGE VS TARGET ===
        print("5. Analyzing age vs target...")
        self.plot_target_vs_numerical(train_df, ['Age'], self.target_col, save=save)
        
        # === EXPENSE PATTERN ANALYSIS ===
        print("6. Analyzing expense patterns...")
        # Only expense-related columns (excluding Age) are used for average expenses analysis
        self.plot_average_expenses_by_target(train_df, self.numerical_cols[1:], self.target_col, save=save) 

        print("\n=== EDA Analysis Completed Successfully! ===\n")

