import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def etl_pipeline(data_path, target_column=None, test_size=0.2, random_state=42):
    """
    Complete ETL pipeline to preprocess, transform and load data for ML

    Parameters:
    - data_path: Path to CSV file
    - target_column: Name of target column (optional)
    - test_size: Proportion of data for test set
    - random_state: Random seed for reproducibility

    Returns:
    - Processed data ready for modeling
    """
    # Extract: Load data
    df = pd.read_csv(data_path)

    # Identify column types
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove target from features if specified
    if target_column:
        if target_column in num_cols:
            num_cols.remove(target_column)
        if target_column in cat_cols:
            cat_cols.remove(target_column)

    # Transform: Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    # Load: Split data and apply transformations
    if target_column:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    else:
        processed_data = preprocessor.fit_transform(df)
        return processed_data, preprocessor

# Example usage:
# X_train, X_test, y_train, y_test, preprocessor = etl_pipeline('data.csv', 'target')