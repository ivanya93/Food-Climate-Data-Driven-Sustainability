#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)
sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px  

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings
# Suppress all warnings for cleaner output in examples
warnings.filterwarnings('ignore')

def get_column_names(data):
    """ This function will be used to extract the column names for numerical and categorical variables
    info from the dataset
    input: dataframe containing all variables
    output: num_vars-> list of numerical columns
            cat_vars -> list of categorical columns"""
        
    num_var = data.select_dtypes(include=['int', 'float']).columns
    print()
    print('Numerical variables are:\n', num_var)
    print('-------------------------------------------------')

    categ_var = data.select_dtypes(include=['category', 'object']).columns
    print('Categorical variables are:\n', categ_var)
    print('-------------------------------------------------') 
    return num_var,categ_var
    
    
def percentage_nullValues(data):
    """
    Function that calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    """
    null_perc = round(data.isnull().sum() / data.shape[0],3) * 100.00
    null_perc = pd.DataFrame(null_perc, columns=['Percentage_NaN'])
    null_perc= null_perc.sort_values(by = ['Percentage_NaN'], ascending = False)
    return null_perc


# In[26]:


def select_threshold(data, thr):
    """
    Function that  calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    
    """
    null_perc = percentage_nullValues(data)
      
    col_keep = null_perc[null_perc['Percentage_NaN'] < thr]
    col_keep = list(col_keep.index)
    print('Columns to keep:',len(col_keep))
    print('Those columns have a percentage of NaN less than', str(thr), ':')
    print(col_keep)
    data_c= data[col_keep]
    
    return data_c


# In[33]:


def fill_na(data):
    """
    Function to fill NaN with mode (categorical variabls) and mean (numerical variables)
    input: data -> df
    """
    for column in data:
        if data[column].dtype != 'object':
            data[column] = data[column].fillna(data[column].mean())  
        else:
            data[column] = data[column].fillna(data[column].mode()[0]) 
    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data


# In[2]:

def OutLiersBox(df,nameOfFeature):
    """
    Function to create a BoxPlot and visualise:
    - All Points in the Variable
    - Suspected Outliers in the variable

    """
    trace0 = go.Box(
        y = df[nameOfFeature],
        name = "All Points",
        jitter = 0.3,
        pointpos = -1.8,
        boxpoints = 'all', #define that we want to plot all points
        marker = dict(
            color = 'rgb(7,40,89)'),
        line = dict(
            color = 'rgb(7,40,89)')
    )

    
    trace1 = go.Box(
        y = df[nameOfFeature],
        name = "Suspected Outliers",
        boxpoints = 'suspectedoutliers', # define the suspected Outliers
        marker = dict(
            color = 'rgba(219, 64, 82, 0.6)',
            #outliercolor = 'rgba(219, 64, 82, 0.6)',
            line = dict(
                outlierwidth = 2)),
        line = dict(
            color = 'rgb(8,81,156)')
    )


    data = [trace0,trace1]

    layout = go.Layout(
        title = "{} Outliers".format(nameOfFeature)
    )

    fig = go.Figure(data=data,layout=layout)
    fig.show()
    #fig.write_html("{}_file.html".format(nameOfFeature))

# In[3]:


def corrCoef(data):
    """
    Function aimed to calculate the corrCoef between each pair of variables

    input: data->dataframe
    """
    data_num = data.select_dtypes(include=['int', 'float'])
    data_corr = data_num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(data_corr,
                xticklabels = data_corr.columns.values,
                yticklabels = data_corr.columns.values,
                annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7))
    plt.title('Correlation Matrix of Numerical Variables')
    plt.show()


# In[4]:

def corrCoef_Threshold(df):
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Draw the heatmap
    sns.heatmap(df.corr(), annot=True, mask = mask, vmax=1,vmin=-1,
                cmap=sns.color_palette("RdBu_r", 7));


def outlier_treatment(df, colname):
    """
    Function that drops the Outliers based on the IQR upper and lower boundaries 
    input: df --> dataframe
           colname --> str, name of the column
    
    """
    
    # Calculate the percentiles and the IQR
    Q1,Q3 = np.percentile(df[colname], [25,75])
    IQR = Q3 - Q1
    
    # Calculate the upper and lower limit
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    
    # Drop the suspected outliers
    df_clean = df[(df[colname] > lower_limit) & (df[colname] < upper_limit)]
    
    print('Shape of the raw data:', df.shape)
    print('..................')
    print('Shape of the cleaned data:', df_clean.shape)
    return df_clean
       
    
def outliers_loop(df_num):
    """
    jsklfjfl
    
    """
    for item in np.arange(0,len(df_num.columns)):
        if item == 0:
            df_c = outlier_treatment(df_num, df_num.columns[item])
        else:
            df_c = outlier_treatment(df_c, df_num.columns[item]) 
    return df_c         

# Define a function to first 20 rows - preview the data
def preview_data(df, num_rows=20):
    """Preview the first few rows of a DataFrame."""
    display(df.head(num_rows))
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    

# Function to find outliers in a group based on the Interquartile Range (IQR) method
def find_outliers(group, col):
    Q1= group[col].quantile(0.25)
    Q3= group[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group[col] < lower_bound) | (group[col] > upper_bound)]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings

# Suppress all warnings for cleaner output in examples
warnings.filterwarnings('ignore')

# --- 1. Exploratory Data Analysis (EDA) Functions ---

def summarize_dataframe(df: pd.DataFrame) -> None:
    """
    Prints a comprehensive summary of a Pandas DataFrame,
    including info, descriptive statistics for numerical and categorical columns,
    and missing value counts.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    print("--- DataFrame Info ---")
    df.info()
    print("\n--- Descriptive Statistics (Numerical) ---")
    print(df.describe())
    print("\n--- Value Counts (Top Categorical) ---")
    for col in df.select_dtypes(include='object').columns:
        print(f"\nValue Counts for '{col}':")
        print(df[col].value_counts(normalize=True).head())
    print("\n--- Missing Values ---")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if not missing_values.empty:
        print(missing_values)
    else:
        print("No missing values found.")


def plot_distributions(df: pd.DataFrame, numerical_cols: list = None, categorical_cols: list = None) -> None:
    """
    Generates histograms for numerical columns and bar plots for categorical columns
    to visualize their distributions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list, optional): List of numerical column names to plot.
                                         If None, all numerical columns are used.
        categorical_cols (list, optional): List of categorical column names to plot.
                                           If None, all object/category columns are used.
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numerical_cols:
        print("\n--- Numerical Column Distributions (Histograms) ---")
        plt.figure(figsize=(15, 5 * ((len(numerical_cols) + 2) // 3)))
        for i, col in enumerate(numerical_cols):
            plt.subplot((len(numerical_cols) + 2) // 3, 3, i + 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    if categorical_cols:
        print("\n--- Categorical Column Distributions (Bar Plots) ---")
        plt.figure(figsize=(15, 5 * ((len(categorical_cols) + 2) // 3)))
        for i, col in enumerate(categorical_cols):
            plt.subplot((len(categorical_cols) + 2) // 3, 3, i + 1)
            sns.countplot(y=df[col].dropna(), order=df[col].value_counts().index)
            plt.title(f'Counts of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson') -> None:
    """
    Generates a correlation heatmap for numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The correlation method ('pearson', 'kendall', 'spearman').
                      Defaults to 'pearson'.
    """
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.empty:
        print("No numerical columns found for correlation heatmap.")
        return

    corr_matrix = numerical_df.corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Heatmap ({method.capitalize()} Method)')
    plt.show()

# --- 2. Statistics Functions ---

def perform_ttest(data1: pd.Series, data2: pd.Series, equal_var: bool = True) -> tuple:
    """
    Performs an independent two-sample t-test on two data series.

    Args:
        data1 (pd.Series): First data series.
        data2 (pd.Series): Second data series.
        equal_var (bool): If True (default), perform a standard independent 2-sample t-test
                          that assumes equal population variances. If False, perform Welch's
                          t-test, which does not assume equal population variance.

    Returns:
        tuple: A tuple containing the t-statistic and the two-tailed p-value.
    """
    t_stat, p_value = stats.ttest_ind(data1.dropna(), data2.dropna(), equal_var=equal_var)
    print(f"Independent Two-Sample T-test (Equal Variances = {equal_var}):")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Result: Statistically significant difference between means (p < 0.05)")
    else:
        print("  Result: No statistically significant difference between means (p >= 0.05)")
    return t_stat, p_value


def calculate_pearson_correlation(series1: pd.Series, series2: pd.Series) -> tuple:
    """
    Calculates the Pearson correlation coefficient and p-value between two series.

    Args:
        series1 (pd.Series): First numerical series.
        series2 (pd.Series): Second numerical series.

    Returns:
        tuple: A tuple containing the Pearson correlation coefficient and the 2-tailed p-value.
    """
    # Drop NaNs to ensure alignment and avoid errors
    combined_df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
    if combined_df.empty:
        print("Cannot calculate correlation: input series are empty after dropping NaNs.")
        return (np.nan, np.nan)

    correlation, p_value = stats.pearsonr(combined_df['s1'], combined_df['s2'])
    print(f"Pearson Correlation between '{series1.name}' and '{series2.name}':")
    print(f"  Correlation Coefficient: {correlation:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Result: Statistically significant linear relationship (p < 0.05)")
    else:
        print("  Result: No statistically significant linear relationship (p >= 0.05)")
    return correlation, p_value

# --- 3. Data Visualization Functions ---

def create_scatterplot(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = None) -> None:
    """
    Generates a scatter plot to show the relationship between two numerical variables.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x (str): Name of the column for the x-axis.
        y (str): Name of the column for the y-axis.
        hue (str, optional): Name of a column to group data points by color. Defaults to None.
        title (str, optional): Title of the plot. Defaults to 'Scatter Plot'.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(title if title else f'Scatter Plot of {y} vs {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def create_boxplot(df: pd.DataFrame, x: str, y: str, title: str = None) -> None:
    """
    Generates a box plot to show the distribution of a numerical variable across categories
    of a categorical variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x (str): Name of the categorical column for the x-axis.
        y (str): Name of the numerical column for the y-axis.
        title (str, optional): Title of the plot. Defaults to 'Box Plot'.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(title if title else f'Box Plot of {y} by {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# --- 4. Data Science (Machine Learning) Functions ---

def preprocess_data(df: pd.DataFrame, numerical_cols: list, categorical_cols: list = None,
                    scaling_method: str = None, test_size: float = 0.2, random_state: int = 42):
    """
    Applies scaling to numerical features and one-hot encoding to categorical features,
    then splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names to be scaled.
        categorical_cols (list, optional): List of categorical column names to be one-hot encoded.
                                           Defaults to None (no categorical encoding).
        scaling_method (str, optional): 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
                                        If None, no scaling is applied. Defaults to None.
        test_size (float): The proportion of the dataset to include in the test split.
                           Defaults to 0.2.
        random_state (int): Controls the shuffling applied to the data before splitting.
                            Defaults to 42.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=['target_column_placeholder'], errors='ignore') # Replace 'target_column_placeholder' with your actual target column name
    y = df['target_column_placeholder'] # Replace 'target_column_placeholder' with your actual target column name

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    if scaling_method == 'standard':
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        print("Numerical columns scaled using StandardScaler.")
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        print("Numerical columns scaled using MinMaxScaler.")
    else:
        print("No scaling applied to numerical columns.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"\nData split into training and testing sets:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def train_evaluate_regression_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Trains a simple Linear Regression model and evaluates its performance.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.

    Returns:
        dict: A dictionary containing 'model', 'predictions', 'mse', 'r2'.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Linear Regression Model Evaluation ---")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")

    return {
        'model': model,
        'predictions': predictions,
        'mse': mse,
        'r2': r2
    }


def train_evaluate_classification_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                        y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Trains a simple Logistic Regression model and evaluates its performance
    with accuracy, classification report, and confusion matrix.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.

    Returns:
        dict: A dictionary containing 'model', 'predictions', 'accuracy',
              'classification_report', 'confusion_matrix'.
    """
    model = LogisticRegression(random_state=42, solver='liblinear') # Using liblinear for robustness with small datasets
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("\n--- Logistic Regression Model Evaluation ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(report)
    print("\n  Confusion Matrix:")
    print(cm)

    return {
        'model': model,
        'predictions': predictions,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Demonstrating Data Science Utility Functions ---")

    # 1. Create a sample DataFrame
    data = {
        'Numerical_Feature_1': np.random.rand(100) * 100,
        'Numerical_Feature_2': np.random.randn(100) * 15 + 50,
        'Categorical_Feature_A': np.random.choice(['Category_X', 'Category_Y', 'Category_Z'], 100),
        'Categorical_Feature_B': np.random.choice(['Group1', 'Group2'], 100),
        'Target_Regression': np.random.rand(100) * 50 + 20,
        'Target_Classification': np.random.randint(0, 2, 100) # Binary target for classification
    }
    # Introduce some missing values
    data['Numerical_Feature_1'][np.random.choice(100, 5, replace=False)] = np.nan
    data['Categorical_Feature_A'][np.random.choice(100, 3, replace=False)] = np.nan

    df = pd.DataFrame(data)
    df['Target_Regression'] = df['Numerical_Feature_1'] * 0.5 + df['Numerical_Feature_2'] * 0.2 + np.random.randn(100) * 5
    df['Target_Classification'] = (df['Numerical_Feature_1'] + df['Numerical_Feature_2'] > 100).astype(int)


    print("\n--- Initial Sample DataFrame Head ---")
    print(df.head())

    # --- EDA Examples ---
    print("\n\n=== EDA Functions Examples ===")
    summarize_dataframe(df)

    numerical_cols_to_plot = ['Numerical_Feature_1', 'Numerical_Feature_2']
    categorical_cols_to_plot = ['Categorical_Feature_A', 'Categorical_Feature_B']
    plot_distributions(df, numerical_cols=numerical_cols_to_plot, categorical_cols=categorical_cols_to_plot)
    plot_correlation_heatmap(df)

    # --- Statistics Examples ---
    print("\n\n=== Statistics Functions Examples ===")
    # Fill NaNs for t-test example to avoid errors
    df_filled = df.fillna(df.mean(numeric_only=True))
    df_filled['Categorical_Feature_A'] = df_filled['Categorical_Feature_A'].fillna(df_filled['Categorical_Feature_A'].mode()[0])

    # T-test: Compare Numerical_Feature_1 between two groups of Categorical_Feature_B
    group1 = df_filled[df_filled['Categorical_Feature_B'] == 'Group1']['Numerical_Feature_1']
    group2 = df_filled[df_filled['Categorical_Feature_B'] == 'Group2']['Numerical_Feature_1']
    if not group1.empty and not group2.empty:
        perform_ttest(group1, group2)
    else:
        print("Cannot perform t-test: one or both groups are empty after filtering.")

    # Pearson Correlation: Between two numerical features
    calculate_pearson_correlation(df['Numerical_Feature_1'], df['Numerical_Feature_2'])

    # --- Data Visualization Examples ---
    print("\n\n=== Data Visualization Functions Examples ===")
    create_scatterplot(df, x='Numerical_Feature_1', y='Numerical_Feature_2', hue='Categorical_Feature_B')
    create_boxplot(df, x='Categorical_Feature_A', y='Numerical_Feature_1')


    # --- Data Science (Machine Learning) Examples ---
    print("\n\n=== Data Science Functions Examples ===")

    # Prepare data for ML functions (handling NaNs for target_column_placeholder)
    # For demonstration, let's drop rows with NaNs in numerical features that will be used.
    # In a real scenario, you'd use imputation.
    ml_df = df.dropna(subset=['Numerical_Feature_1', 'Numerical_Feature_2', 'Target_Regression', 'Target_Classification', 'Categorical_Feature_A', 'Categorical_Feature_B'])
    
    # Rename target column for demonstration purposes if needed, otherwise skip
    ml_df = ml_df.rename(columns={'Target_Regression': 'target_column_placeholder_regression',
                                  'Target_Classification': 'target_column_placeholder_classification'})

    # Example for Regression
    print("\n--- Regression Model Demo ---")
    # Temporarily set 'target_column_placeholder' for the preprocess_data function
    temp_df_regression = ml_df.copy()
    temp_df_regression['target_column_placeholder'] = temp_df_regression['target_column_placeholder_regression']

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocess_data(
        temp_df_regression,
        numerical_cols=['Numerical_Feature_1', 'Numerical_Feature_2'],
        categorical_cols=['Categorical_Feature_A', 'Categorical_Feature_B'],
        scaling_method='standard'
    )
    regression_results = train_evaluate_regression_model(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    print(f"Regression Model (LinearRegression) R2: {regression_results['r2']:.4f}")

    # Example for Classification
    print("\n--- Classification Model Demo ---")
    # Temporarily set 'target_column_placeholder' for the preprocess_data function
    temp_df_classification = ml_df.copy()
    temp_df_classification['target_column_placeholder'] = temp_df_classification['target_column_placeholder_classification']

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = preprocess_data(
        temp_df_classification,
        numerical_cols=['Numerical_Feature_1', 'Numerical_Feature_2'],
        categorical_cols=['Categorical_Feature_A', 'Categorical_Feature_B'],
        scaling_method='minmax'
    )
    classification_results = train_evaluate_classification_model(X_train_cls, X_test_cls, y_train_cls, y_test_cls)
    print(f"Classification Model (LogisticRegression) Accuracy: {classification_results['accuracy']:.4f}")
