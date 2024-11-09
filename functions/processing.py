import os
import pandas as pd
import numpy as np
import logging
from scipy import stats
import shutil
import schedule
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(filename='data_cleaning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def handle_missing_values(df, threshold=0.3, strategy='auto', fill_value=None):
    """
    Automatically handle missing values based on column type and distribution.
    For numerical columns: mean, median, or mode.
    For categorical columns: mode.
    """
    try:
        for column in df.columns:
            missing_percentage = df[column].isnull().mean()

            # Skip if there are no missing values
            if missing_percentage == 0:
                continue

            # If the missing values are above the threshold, log and skip processing
            if missing_percentage > threshold:
                logging.info(f"Skipping column '{column}' due to excessive missing values ({missing_percentage*100}%)")
                continue

            if df[column].dtype == 'object':  # Categorical columns
                if strategy == 'auto' or strategy == 'mode':
                    df[column] = df[column].fillna(df[column].mode().iloc[0])  # Fill with mode
                    logging.info(f"Handled missing values in column '{column}' using mode.")
                else:
                    logging.warning(f"Unsupported strategy '{strategy}' for categorical data in column '{column}'")
            
            else:  # Numerical columns
                # Check for normal distribution using skewness
                skewness = df[column].skew()

                if strategy == 'auto':
                    if abs(skewness) < 0.5:
                        df[column] = df[column].fillna(df[column].mean())
                        logging.info(f"Handled missing values in column '{column}' using mean (normal distribution).")
                    else:
                        df[column] = df[column].fillna(df[column].median())
                        logging.info(f"Handled missing values in column '{column}' using median (skewed distribution).")
                elif strategy == 'mean':
                    df[column] = df[column].fillna(df[column].mean())
                    logging.info(f"Handled missing values in column '{column}' using mean.")
                elif strategy == 'median':
                    df[column] = df[column].fillna(df[column].median())
                    logging.info(f"Handled missing values in column '{column}' using median.")
                elif strategy == 'mode':
                    df[column] = df[column].fillna(df[column].mode().iloc[0])
                    logging.info(f"Handled missing values in column '{column}' using mode.")
                elif strategy == 'custom' and fill_value is not None:
                    df[column] = df[column].fillna(fill_value)
                    logging.info(f"Handled missing values in column '{column}' using custom value.")
                else:
                    logging.warning(f"Unsupported strategy '{strategy}' for numerical data in column '{column}'")
        
    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
    return df

# Function to format column names
def format_columns(df):
    try:
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        logging.info("Formatted column names.")
    except Exception as e:
        logging.error(f"Error formatting columns: {e}")
    return df

# Function to handle outliers
def handle_outliers(df, z_thresh=3):
    try:
        df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < z_thresh).all(axis=1)]
        logging.info("Removed outliers based on Z-score threshold.")
    except Exception as e:
        logging.error(f"Error handling outliers: {e}")
    return df

# Function to plot the difference between raw and cleaned datasets
def plot_data_comparison(raw_df, cleaned_df):
    try:
        raw_missing = raw_df.isnull().mean() * 100
        cleaned_missing = cleaned_df.isnull().mean() * 100

        # Save the missing values comparison plot as an image
        plt.figure(figsize=(10, 6))
        sns.barplot(x=raw_missing.index, y=raw_missing, color='red', label='Raw Data')
        sns.barplot(x=cleaned_missing.index, y=cleaned_missing, color='green', label='Cleaned Data')
        plt.xticks(rotation=90)
        plt.xlabel('Columns')
        plt.ylabel('Percentage of Missing Values')
        plt.title('Missing Values Comparison (Raw vs Cleaned)')
        plt.legend()
        plot_path = 'static/missing_values_comparison.png'
        plt.savefig(plot_path)
        plt.close()

        # Now save individual numerical column distributions
        numerical_columns = raw_df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(raw_df[column], kde=True, color='red', label='Raw Data', alpha=0.6)
            sns.histplot(cleaned_df[column], kde=True, color='green', label='Cleaned Data', alpha=0.6)
            plt.title(f'Distribution of {column} (Raw vs Cleaned)')
            plt.legend()
            plot_path = f'static/{column}_distribution.png'
            plt.savefig(plot_path)
            plt.close()

    except Exception as e:
        logging.error(f"Error plotting data comparison: {e}")

# Function for the data cleaning pipeline
def clean_data_pipeline(file_path, output_folder):
    try:
        if not os.path.exists(file_path):
            logging.error(f"Input folder does not exist: {file_path}")
            return None

        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}")

        df = format_columns(df)
        df = handle_missing_values(df, strategy='mean')
        df = handle_outliers(df, z_thresh=3)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logging.info(f"Output folder created: {output_folder}")

        cleaned_file_path = os.path.join(output_folder, "cleaned_" + os.path.basename(file_path))
        df.to_csv(cleaned_file_path, index=False)
        logging.info(f"Saved cleaned data to {cleaned_file_path}")

        raw_df = pd.read_csv(file_path)
        plot_data_comparison(raw_df, df)

    except Exception as e:
        logging.error(f"Error in cleaning pipeline for {file_path}: {e}")
    return df

# Function to check if the dataset is good to use (using a simple model)
def check_dataset_quality(df, model=None):
    try:
        if df.isnull().sum().sum() > 0:
            logging.info("Dataset has missing values. It might need further cleaning.")
        if len(df) < 10:
            logging.info("Dataset is too small for analysis.")
        
        X = df.dropna(axis=1, how='any')  # Drop columns with any missing values
        if X.shape[1] < 1:
            logging.info("Dataset has no valid features to use.")
            return False

        # If you have a model passed, test its accuracy
        if model is not None:
            y = df.iloc[:, -1]  # Assume the last column is the target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Accuracy of model on this dataset: {accuracy:.2f}")
            return accuracy > 0.6  # Consider dataset "good" if accuracy > 0.6
        
        return True

    except Exception as e:
        logging.error(f"Error in dataset quality check: {e}")
        return False

# Function to calculate and display the confusion matrix
def generate_confusion_matrix(df, model=None):
    try:
        if model is not None:
            X = df.dropna(axis=1, how='any')
            y = df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.show()

    except Exception as e:
        logging.error(f"Error in generating confusion matrix: {e}")

# Batch process and schedule task
def batch_process_data(input_folder, output_folder):
    try:
        if not os.path.exists(input_folder):
            logging.error(f"Input folder does not exist: {input_folder}")
            return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logging.info(f"Output folder created: {output_folder}")

        for filename in os.listdir(input_folder):
            if filename.endswith('.csv') or filename.endswith('xlsx'):
                file_path = os.path.join(input_folder, filename)
                cleaned_df = clean_data_pipeline(file_path, output_folder)

                # Check if cleaned dataset is good to use
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                if cleaned_df is not None and check_dataset_quality(cleaned_df, model=model):
                    logging.info(f"Dataset {filename} is good to use.")
                    generate_confusion_matrix(cleaned_df, model=model)
                else:
                    logging.info(f"Dataset {filename} might not be usable for modeling.")

    except Exception as e:
        logging.error(f"Error in batch processing: {e}")

# Schedule batch processing to run periodically
def schedule_batch_processing():
    try:
        input_folder = "test"  # Input folder with raw data
        output_folder = "cleaned_data"  # Output folder for cleaned data
        schedule.every().day.at("01:00").do(batch_process_data, input_folder=input_folder, output_folder=output_folder)

        logging.info("Batch processing scheduled to run daily at 01:00 AM.")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Wait for 1 minute before checking again

    except Exception as e:
        logging.error(f"Error scheduling batch processing: {e}")
