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
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import pickle
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
    """Cleans time series data by removing duplicates, NaNs, and setting datetime index."""
    def __init__(self, column: str):
        super().__init__()
        self.date_column = column

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data")
        initial_shape = data.shape

        data = data.drop_duplicates()
        data = data.dropna()

        if self.date_column in data.columns:
            try:
                data[self.date_column] = pd.to_datetime(data[self.date_column], dayfirst=True)
                data["Date"] = data[self.date_column]
            except Exception as e:
                logger.error(f"Failed to convert '{self.date_column}' to datetime", exc_info=True)
                raise

            data = data.sort_values(self.date_column, ascending=True)
            data = data.set_index(self.date_column)

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
                data['year'] = data[self.date_column].dt.year
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
                period = data[self.column].max() or 1

        logger.info(f"Creating cyclic features for {self.column} with period {period}")
        try:
            data[f"{self.column}_sin"] = np.sin(2 * np.pi * data[self.column] / period)
            data[f"{self.column}_cos"] = np.cos(2 * np.pi * data[self.column] / period)
        except Exception as e:
            logger.error(f"Error creating cyclic features: {e}")
        return data


class IFeatureScaler:
    """Interface for scaling of data"""
    def scale(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        raise NotImplementedError


class StandardScalerFeatureScaler(IFeatureScaler):
    """Scales the numerical columns using StandardScaler"""
    def __init__(self, column: str):
        self.scaler = StandardScaler()
        self.column = column
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        if self.column in data.columns:
            self.scaler.fit(data[[self.column]])
            self.fitted = True
            logger.info(f"Fitted scaler on column {self.column}")
        else:
            logger.warning(f"Column '{self.column}' not found for fitting scaler.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            logger.error("Scaler has not been fitted yet.")
            return data
        if self.column in data.columns:
            data[self.column] = self.scaler.transform(data[[self.column]])
            logger.info(f"Transformed column {self.column}")
        else:
            logger.warning(f"Column '{self.column}' not found for scaling.")
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column in data.columns:
            data[self.column] = self.scaler.fit_transform(data[[self.column]])
            self.fitted = True
            logger.info(f"Saving StandardScaler file ...")
            scalers_dir = (Path(__file__).parent / "../models/scalers").resolve()
            scalers_dir.mkdir(parents=True, exist_ok=True)

            scaler_file = scalers_dir / f"{self.column}_scaler.pkl"
            with open(scaler_file, "wb") as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Saved standardscaler for {self.column} to {scaler_file}")
            logger.info(f"Fit-transformed column {self.column}")
        else:
            logger.warning(f"Column '{self.column}' not found for scaling.")
        return data

    def scale(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if fit:
            data = data.copy()
            return self.fit_transform(data)
        else:
            return self.transform(data)


class ILabelEncoder:
    """Inteface for label encoding."""
    def encode(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        raise NotImplementedError


class LabelEncoderFeature(ILabelEncoder):
    """Encodes Categorical Features using LabelEncoder"""
    def __init__(self, column: str):
        self.column = column
        self.encoder = LabelEncoder()
        self.fitted = False
    
    def fit(self, data: pd.DataFrame) -> None:
        if self.column in data.columns:
            self.encoder.fit(data[self.column])
            self.fitted = True
            logger.info(f"Fitted label encoder on column {self.column}")
        else:
            logger.warning(f"Column '{self.column}' not found for fitting label encoder.")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            logger.error("Label encoder has not been fitted yet.")
            return data

        if self.column in data.columns:
            # Map known labels, unseen labels get -1
            classes = set(self.encoder.classes_)
            data[self.column] = data[self.column].apply(
                lambda x: self.encoder.transform([x])[0] if x in classes else -1
            )
            logger.info(f"Transformed column {self.column} (unseen labels mapped to -1)")
        else:
            logger.warning(f"Column '{self.column}' not found for encoding.")
        return data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms the data using LabelEncoder"""
        if self.column in data.columns:
            data[self.column] = self.encoder.fit_transform(data[self.column])
            self.fitted = True
            logger.info(f"Fit-transformed column {self.column}")
        else:
            logger.warning(f"Column '{self.column}' not found for encoding.")
        return data

    def encode(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encodes the data using LabelEncoder"""
        if fit:
            data = data.copy()
            data = self.fit_transform(data)
            mappings = {str(cls): int(idx) for idx, cls in enumerate(self.encoder.classes_)}
            mapping_dir = (Path(__file__).parent / '../data/mappings').resolve()
            mapping_dir.mkdir(parents=True, exist_ok=True)
            mapping_file = mapping_dir / f"{self.column}_mapping.json"

            encoder_dir = (Path(__file__).parent / '../models/encoders').resolve()
            encoder_dir.mkdir(parents=True, exist_ok=True)
            encoder_file = encoder_dir / f"{self.column}_encoder.pkl"

            with open(mapping_file, 'w') as f:
                json.dump(mappings, f)
            
            with open(encoder_file, "wb") as f:
                pickle.dump(self.encoder, f)

            logger.info(f"Saved mapping for {self.column} to {mapping_file} and encoder to {encoder_file}")
            return data
        else:
            return self.transform(data)


class IDataSplitter:
    """Interface for data splitting."""
    def split(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


class TrainTestSplitter(IDataSplitter):
    """Splits data into train and test sets."""
    def split(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        logger.info(f"Splitting data into train and test sets with test size {test_size}")
        try:
            data_len: int = len(data)
            test_len: int = int(data_len * test_size)
            
            logger.info(f"Splitting time series data into train and test Splits with test length {test_len}")
            train_set = data.iloc[:-test_len]
            test_set = data.iloc[-test_len:]

            result = {
                "train": train_set,
                "test": test_set
            }
            return result

        except Exception as e:
            logger.error(f"Failed to split the data to Train and test sets")
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
        scaler: IFeatureScaler,
        encoder: List[ILabelEncoder],
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
                if split_name == 'train':
                    split_data = scaler.scale(split_data, fit=True)
                    for encoder in encoders:
                        split_data = encoder.encode(split_data, fit=True)
                else:
                    split_data = scaler.scale(split_data, fit=False)
                    for encoder in encoders:
                        split_data = encoder.encode(split_data, fit=False)

                split_file = os.path.join(self.output_dir, f"{split_name}.csv")
                self.saver.save(split_data, split_file)
            return data
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise


if __name__ == "__main__":
    # Define input and output paths
    argparser = argparse.ArgumentParser(description="Data Preprocessing Script for Time Series Forecasting")
    argparser.add_argument(
        "--input_file",
        type=str,
        default='data/train.csv',
        help="Path to the input CSV file, e.g 'data/train.csv'"
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default='data/processed',
        help="Directory to save the processed data, e.g 'data/processed'"
    )
    args = argparser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    # List of required columns
    cols = [
        'Sub-Category',
        'dayofweek_cos',
        'dayofweek_sin',
        'Customer Name',
        'State',
        'year_cos',
        'year_sin',
        'City',
        'month_cos',
        'month_sin',
        'Product Name',
        'Sales'
    ]

    # Compose pipeline
    loader = CSVDataLoader(input_file)
    cleaner = BasicDataCleaner("Ship Date")
    feature_engineers = [
        DateTimeFeatureEngineer('Date'),
        CyclicFeatureEngineer('dayofweek'),
        CyclicFeatureEngineer('month'),
        CyclicFeatureEngineer('year')
    ]
    column_selector = ColumnSelector(cols)
    splitter = TrainTestSplitter()
    scaler = StandardScalerFeatureScaler('Sales')
    encoders = [
        LabelEncoderFeature('Sub-Category'),
        LabelEncoderFeature('Product Name'),
        LabelEncoderFeature('Customer Name'),
        LabelEncoderFeature('State'),
        LabelEncoderFeature('City')
    ]
    saver = CSVDataSaver()

    preprocessor = DataPreprocessor(
        loader=loader,
        cleaner=cleaner,
        feature_engineers=feature_engineers,
        column_selector=column_selector,
        splitter=splitter,
        scaler=scaler,
        encoder=encoders,
        saver=saver,
        output_dir=output_dir
    )

    preprocessor.run()
