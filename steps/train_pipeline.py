#!/usr/bin/env python3

"""
AUTHOR: Dan Njuguna
DATE: 2025-05-14

DESCRIPTION: This script implements the training loop logic for the Time Series Forecasting
    model. It is descigned to work in largescale and train on workloads of data above 1000+
    rows when retraining.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Ensure logs directory exists
os.makedirs('../logs', exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_file_path = Path(__file__).parent / '../logs/train_pipeline.log'
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Set these based on your data shape
INPUT_DIM = 24
OUTPUT_DIM = 3
HIDDEN_DIM = 32
BATCH_SIZE = 16


# TODO: Implement Pytorch Dataset class
class TimeSeriesDataset(Dataset):
    """Custom dataset for training the time series forecasting model.
    -----
    Args:
        data (numpy.ndarray): The input time series data.
        input_window (int): the number of previous time steps as input.
        output_window (int): the number of future time steps to predict
        target_indices (list, optional): list of the target variables. (default: all).
    ------
    Returns:
        torch.Tensor: The input time series data for the model.
        torch.Tensor: The target time series data for the model.
    """
    def __init__(self, data, input_window, output_window, target_indices=None):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.target_indices = target_indices or list(range(len(data[0])))
        self.length = len(data) - input_window - output_window + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_window]
        y = self.data[idx + self.input_window: idx + self.input_window + self.output_window, self.target_indices]
        # Flatten y to match model output shape if output_window > 1
        y = y.flatten()
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TimeSeriesModel(nn.Module):
    """LSTM model for time series forecasting.
    -----
    Args:
        input_size (int): The number of features in the input data.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of features in the output data.
        num_layers (int): The number of recurrent layers. (default: 1)
    ------
    Returns:
        torch.Tensor: The output of the model.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """Train the model.
    -----
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_epochs (int): The number of epochs to train for. (default: 10)
    ------
    Returns:
        None
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="green") as pbar:
            for inputs, targets in pbar:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        logger.info(f"Model saved at epoch {epoch + 1}")


if __name__ == "__main__":
    # TODO: Load pandas dataset
    csv_path = Path(__file__).parent / '../data/processed/train.csv'

    try:
        df = pd.read_csv(csv_path)
        data = df.values
        logger.info(f"Data loaded successfully from {csv_path}")

        input_window = INPUT_DIM
        output_window = OUTPUT_DIM
        target_cols = [1, -1]

        dataset = TimeSeriesDataset(data, input_window, output_window, target_indices=target_cols)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        logger.info(f"Data loaded successfully from {csv_path}")

        model = TimeSeriesModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, num_epochs=10)
        logger.info("Training completed successfully")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        logger.info("Training pipeline finished")
        # Ensure models directory exists before saving
        os.makedirs('../models', exist_ok=True)
        torch.save(model.state_dict(), '../models/time_series_model.pth')
        logger.info("Model saved successfully")
        logger.info("Model saved successfully")
