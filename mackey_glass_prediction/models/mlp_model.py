import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    MLP (Multi-Layer Perceptron) model for time series prediction
    """
    
    def __init__(self, input_size=20, hidden_sizes=[128, 64, 32], output_size=1, 
                 dropout_rate=0.2, activation='relu'):
        """
        Parameters:
        - input_size: input window size
        - hidden_sizes: list with the number of neurons in each hidden layer
        - output_size: number of steps ahead to predict
        - dropout_rate: dropout rate for regularization
        - activation: activation function ('relu', 'tanh', 'leaky_relu')
        """
        super(MLPModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Create network layers
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(self._get_activation())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self):
        """
        Returns the chosen activation function
        """
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()  # default
    
    def _initialize_weights(self):
        """
        Initialize network weights
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass of the network
        """
        # Flatten input if necessary
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def get_model_info(self):
        """
        Returns information about the model
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'MLP',
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
        
        return info
    
    def print_model_summary(self):
        """
        Print a model summary
        """
        info = self.get_model_info()
        print("=" * 50)
        print("MLP MODEL SUMMARY")
        print("=" * 50)
        print(f"Type: {info['model_type']}")
        print(f"Input size: {info['input_size']}")
        print(f"Hidden layers: {info['hidden_sizes']}")
        print(f"Output size: {info['output_size']}")
        print(f"Dropout rate: {info['dropout_rate']}")
        print(f"Activation function: {info['activation']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        print("=" * 50)


def create_mlp_variants():
    """
    Create different MLP model variations for experimentation
    """
    variants = {
        'mlp_small': MLPModel(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1,
            dropout_rate=0.1,
            activation='relu'
        ),
        'mlp_medium': MLPModel(
            input_size=20,
            hidden_sizes=[128, 64, 32],
            output_size=1,
            dropout_rate=0.2,
            activation='relu'
        ),
        'mlp_large': MLPModel(
            input_size=20,
            hidden_sizes=[256, 128, 64, 32],
            output_size=1,
            dropout_rate=0.3,
            activation='relu'
        ),
        'mlp_tanh': MLPModel(
            input_size=20,
            hidden_sizes=[128, 64, 32],
            output_size=1,
            dropout_rate=0.2,
            activation='tanh'
        ),
        'mlp_leaky': MLPModel(
            input_size=20,
            hidden_sizes=[128, 64, 32],
            output_size=1,
            dropout_rate=0.2,
            activation='leaky_relu'
        )
    }
    
    return variants 