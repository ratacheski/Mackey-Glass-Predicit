import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd


class MackeyGlassGenerator:
    """
    Gerador da série temporal de Mackey-Glass
    """
    
    def __init__(self, n_points=10000, tau=20, gamma=0.2, beta=0.4, n=18, x0=1):
        """
        Parâmetros:
        - n_points: número de pontos a gerar
        - tau: delay (padrão 20)
        - gamma: constante (padrão 0.2) 
        - beta: constante (padrão 0.4)
        - n: constante (padrão 18)
        - x0: valor inicial (padrão 1)
        """
        self.n_points = n_points
        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.n = n
        self.x0 = x0
        
    def generate_series(self):
        """
        Gera a série temporal de Mackey-Glass usando equação diferencial
        """
        # Inicializar array
        x = np.zeros(self.n_points + self.tau)
        
        # Condições iniciais
        x[0] = self.x0
        
        # Gerar série usando equação de Mackey-Glass
        for i in range(1, self.n_points + self.tau):
            if i <= self.tau:
                x[i] = x[i-1] + (self.beta * x[0] / (1 + x[0]**self.n) - self.gamma * x[i-1])
            else:
                x[i] = x[i-1] + (self.beta * x[i-self.tau] / (1 + x[i-self.tau]**self.n) - self.gamma * x[i-1])
        
        # Retornar apenas os pontos válidos (depois do delay)
        return x[self.tau:]
    
    def save_series(self, series, filename):
        """
        Salva a série em arquivo CSV
        """
        df = pd.DataFrame({'time': range(len(series)), 'value': series})
        df.to_csv(filename, index=False)
        print(f"Série salva em {filename}")
    
    def plot_series(self, series, title="Série Temporal de Mackey-Glass", save_path=None):
        """
        Plota a série temporal
        """
        plt.figure(figsize=(12, 6))
        plt.plot(series)
        plt.title(title)
        plt.xlabel('Tempo')
        plt.ylabel('Valor')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Gráfico salvo em {save_path}")
        
        plt.show()


class MackeyGlassDataset(Dataset):
    """
    Dataset personalizado para série temporal de Mackey-Glass
    """
    
    def __init__(self, series, window_size=20, prediction_steps=1):
        """
        Parâmetros:
        - series: série temporal
        - window_size: tamanho da janela de entrada
        - prediction_steps: número de passos à frente para predizer
        """
        self.series = series
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        
        # Normalizar série
        self.mean = np.mean(series)
        self.std = np.std(series)
        self.normalized_series = (series - self.mean) / self.std
        
        # Criar janelas de dados
        self.X, self.y = self._create_windows()
    
    def _create_windows(self):
        """
        Cria janelas de dados para treinamento
        """
        X, y = [], []
        
        for i in range(len(self.normalized_series) - self.window_size - self.prediction_steps + 1):
            # Janela de entrada
            X.append(self.normalized_series[i:i + self.window_size])
            # Valor(es) a predizer
            y.append(self.normalized_series[i + self.window_size:i + self.window_size + self.prediction_steps])
        
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
    
    def denormalize(self, normalized_value):
        """
        Desnormaliza valores preditos
        """
        return normalized_value * self.std + self.mean
    
    def get_train_test_split(self, train_ratio=0.9):
        """
        Divide os dados em treino e teste
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
    Cria DataLoaders para treino e teste
    """
    dataset = MackeyGlassDataset(series, window_size, prediction_steps)
    (train_X, train_y), (test_X, test_y) = dataset.get_train_test_split(train_ratio)
    
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset 