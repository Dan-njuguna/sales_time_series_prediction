#!/usr/bin/env python3
"""
AUTHOR: Dan Njuguna
DATE: 2025-05-13

DESCRIPTION: This script implements all the data preprocessing steps: cleaning, splitting, scaling,
    encoding and saving the data. It uses the `pandas` library for data manipulation and `scikit-learn`
    for scaling and encoding. The script is designed to be run from the command line and takes
    command line arguments for the input and output file paths. The script also includes a function
    to load the data from a CSV file, clean it by removing duplicates and null values, split it into
    training and testing sets, scale the features using StandardScaler, and encode categorical variables
    using LabelEncoder.

    The encoding mapping is saved to JSON files for later use in the model training and evaluation steps.
    The script also includes a function to save the preprocessed data to a CSV file.
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Ensure logs directory exists
os.makedirs('../logs', exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_file_path = Path(__file__).parent / '../logs/preprocess_data.log'
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class IDataLoader:
    """Interface for data loading."""
    def load(self) -> pd.DataFrame:
        raise NotImplementedError


class CSVDataLoader(IDataLoader):
    """Loads data from a CSV file."""
    def __init__(self, file_path: str):
        self.file_path = Path(file_path).resolve()

    def load(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {self.file_path}")
            data = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise


class IDataCleaner:
    """Interface for data cleaning."""
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class BasicDataCleaner(IDataCleaner):
    """Removes duplicates and null values."""
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data")
        initial_shape = data.shape
        data = data.drop_duplicates()
        data = data.dropna()
        logger.info(f"Data cleaned: {initial_shape} -> {data.shape}")
        return data


class IFeatureEngineer:
    """Interface for feature engineering."""
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class DateTimeFeatureEngineer(IFeatureEngineer):
    """Adds datetime-based features."""
    def __init__(self, date_column: str = 'Ship Date'):
        self.date_column = date_column

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating datetime features")
        if self.date_column in data.columns:
            try:
                data[self.date_column] = pd.to_datetime(data[self.date_column], dayfirst=True)
                data['dayofweek'] = data[self.date_column].dt.dayofweek
                data['month'] = data[self.date_column].dt.month
            except Exception as e:
                logger.error(f"Error creating datetime features: {e}")
        else:
            logger.warning(f"No '{self.date_column}' column found to create datetime features.")
        return data


class CyclicFeatureEngineer(IFeatureEngineer):
    """Adds cyclic (sine and cosine) features for a time-related column."""
    def __init__(self, column: str, period: Optional[int] = None):
        self.column = column
        self.period = period

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            logger.warning(f"Column '{self.column}' not found for cyclic feature engineering.")
            return data

        period = self.period
        if period is None:
            if self.column == 'dayofweek':
                period = 7
            elif self.column == 'month':
                period = 12
            else:
                period = data[self.column].max() or 1  # avoid division by zero

        logger.info(f"Creating cyclic features for {self.column} with period {period}")
        try:
            data[f"{self.column}_sin"] = np.sin(2 * np.pi * data[self.column] / period)
            data[f"{self.column}_cos"] = np.cos(2 * np.pi * data[self.column] / period)
        except Exception as e:
            logger.error(f"Error creating cyclic features: {e}")
        return data


class IDataSplitter:
    """Interface for data splitting."""
    def split(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


class TrainValTestSplitter(IDataSplitter):
    """Splits data into train, validation, and test sets."""
    def split(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        logger.info(f"Splitting data into train, val, and test sets with test size {test_size}")
        try:
            train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
            train_data, val_data = train_test_split(train_data, test_size=test_size, random_state=42)
            logger.info(f"Train shape: {train_data.shape}, Validation shape: {val_data.shape}, Test shape: {test_data.shape}")
            return {'train': train_data, 'val': val_data, 'test': test_data}
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise


class IDataSaver:
    """Interface for saving data."""
    def save(self, data: pd.DataFrame, file_path: str) -> None:
        raise NotImplementedError


class CSVDataSaver(IDataSaver):
    """Saves DataFrame to a CSV file."""
    def save(self, data: pd.DataFrame, file_path: str) -> None:
        try:
            output_path = Path(file_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")
            raise RuntimeError(f"Failed to save data to {file_path}: {e}")


class ColumnSelector:
    """Keeps only specified columns in the DataFrame."""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def select(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Keeping necessary columns: {self.columns}")
        try:
            data = data[self.columns]
            logger.info(f"Data shape after removing columns: {data.shape}")
        except Exception as e:
            logger.error(f"Error selecting columns: {e}")
            raise
        return data


class DataPreprocessor:
    """Coordinates the data preprocessing pipeline."""

    def __init__(
        self,
        loader: IDataLoader,
        cleaner: IDataCleaner,
        feature_engineers: List[IFeatureEngineer],
        column_selector: ColumnSelector,
        splitter: IDataSplitter,
        saver: IDataSaver,
        output_dir: str
    ):
        self.loader = loader
        self.cleaner = cleaner
        self.feature_engineers = feature_engineers
        self.column_selector = column_selector
        self.splitter = splitter
        self.saver = saver
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> pd.DataFrame:
        try:
            data = self.loader.load()
            data = self.cleaner.clean(data)
            for engineer in self.feature_engineers:
                data = engineer.add_features(data)
            data = self.column_selector.select(data)
            output_file = os.path.join(self.output_dir, 'preprocessed_sales_data.csv')
            self.saver.save(data, output_file)
            logger.info(f"Head of the preprocessed data:\n{data.head()}")
            splits = self.splitter.split(data)
            for split_name, split_data in splits.items():
                split_file = os.path.join(self.output_dir, f"{split_name}.csv")
                self.saver.save(split_data, split_file)
            return data
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise


if __name__ == "__main__":
    # Define input and output paths
    input_file = 'data/train.csv'
    output_dir = 'data/processed'

    # List of required columns
    cols = [
        'Sub-Category',
        'Product Name',
        'dayofweek_cos',
        'Customer Name',
        'State',
        'City',
        'month_cos',
        'Sales'
    ]

    # Compose pipeline
    loader = CSVDataLoader(input_file)
    cleaner = BasicDataCleaner()
    feature_engineers = [
        DateTimeFeatureEngineer('Ship Date'),
        CyclicFeatureEngineer('dayofweek'),
        CyclicFeatureEngineer('month')
    ]
    column_selector = ColumnSelector(cols)
    splitter = TrainValTestSplitter()
    saver = CSVDataSaver()

    preprocessor = DataPreprocessor(
        loader=loader,
        cleaner=cleaner,
        feature_engineers=feature_engineers,
        column_selector=column_selector,
        splitter=splitter,
        saver=saver,
        output_dir=output_dir
    )

    preprocessor.run()
