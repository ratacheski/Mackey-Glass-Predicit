import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MackeyGlassGenerator:
    """
    Mackey-Glass time series generator
    """
    
    def __init__(self, n_points=10000, tau=20, gamma=0.2, beta=0.4, n=18, x0=1):
        """
        Parameters:
        - n_points: number of points to generate
        - tau: delay (default 20)
        - gamma: constant (default 0.2) 
        - beta: constant (default 0.4)
        - n: constant (default 18)
        - x0: initial value (default 1)
        """
        self.n_points = n_points
        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.n = n
        self.x0 = x0
        
    def generate_series(self):
        """
        Generate Mackey-Glass time series using differential equation
        """
        # Initialize array
        x = np.zeros(self.n_points + self.tau)
        
        # Initial conditions
        x[0] = self.x0
        
        # Generate series using Mackey-Glass equation
        for i in range(1, self.n_points + self.tau):
            if i <= self.tau:
                x[i] = x[i-1] + (self.beta * x[0] / (1 + x[0]**self.n) - self.gamma * x[i-1])
            else:
                x[i] = x[i-1] + (self.beta * x[i-self.tau] / (1 + x[i-self.tau]**self.n) - self.gamma * x[i-1])
        
        # Return only valid points (after delay)
        return x[self.tau:]
    
    def save_series(self, series, filename):
        """
        Save series to CSV file
        """
        df = pd.DataFrame({'time': range(len(series)), 'value': series})
        df.to_csv(filename, index=False)
        print(f"Series saved to {filename}")

class MackeyGlassDataset(Dataset):
    """
    Custom dataset for Mackey-Glass time series
    """
    
    def __init__(self, series, window_size=20, prediction_steps=1):
        """
        Parameters:
        - series: time series
        - window_size: input window size
        - prediction_steps: number of steps ahead to predict
        """
        self.series = series
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        
        # Normalize series
        self.mean = np.mean(series)
        self.std = np.std(series)
        self.normalized_series = (series - self.mean) / self.std
        
        # Create data windows
        self.X, self.y = self._create_windows()
    
    def _create_windows(self):
        """
        Create data windows for training
        """
        X, y = [], []
        
        for i in range(len(self.normalized_series) - self.window_size - self.prediction_steps + 1):
            # Input window
            X.append(self.normalized_series[i:i + self.window_size])
            # Value(s) to predict
            y.append(self.normalized_series[i + self.window_size:i + self.window_size + self.prediction_steps])
        
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
    
    def denormalize(self, normalized_value):
        """
        Denormalize predicted values
        """
        return normalized_value * self.std + self.mean
    
    def get_train_test_split(self, train_ratio=0.9):
        """
        Split data into training and test sets
        """
        train_size = int(len(self) * train_ratio)
        
        train_X = self.X[:train_size]
        train_y = self.y[:train_size]
        test_X = self.X[train_size:]
        test_y = self.y[train_size:]
        
        return (torch.FloatTensor(train_X), torch.FloatTensor(train_y)), \
               (torch.FloatTensor(test_X), torch.FloatTensor(test_y))


def create_dataloaders(series, window_size=20, prediction_steps=1, 
                      train_ratio=0.9, batch_size=32, shuffle=True):
    """
    Create DataLoaders for training and testing
    """
    dataset = MackeyGlassDataset(series, window_size, prediction_steps)
    (train_X, train_y), (test_X, test_y) = dataset.get_train_test_split(train_ratio)
    
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset 