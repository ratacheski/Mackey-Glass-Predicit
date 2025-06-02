import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    LSTM (Long Short-Term Memory) model for time series prediction
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1,
                 dropout_rate=0.2, bidirectional=False, use_attention=False):
        """
        Parameters:
        - input_size: number of input features (1 for univariate series)
        - hidden_size: number of LSTM units in each layer
        - num_layers: number of stacked LSTM layers
        - output_size: number of steps ahead to predict
        - dropout_rate: dropout rate for regularization
        - bidirectional: if True, uses bidirectional LSTM
        - use_attention: if True, adds attention mechanism
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention layer (optional)
        if use_attention:
            self.attention = AttentionLayer(lstm_output_size)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass of the network
        """
        batch_size = x.size(0)
        
        # Reshape to (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Add feature dimension
        
        # Initialize hidden states
        h0, c0 = self._init_hidden(batch_size, x.device)
        
        # Forward pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
        
        # Apply attention if configured
        if self.use_attention:
            lstm_out = self.attention(lstm_out)
        else:
            # Use only the last output of the sequence
            lstm_out = lstm_out[:, -1, :]
        
        # Forward pass through fully connected layers
        output = self.fc_layers(lstm_out)
        
        return output
    
    def _init_hidden(self, batch_size, device):
        """
        Initialize LSTM hidden states
        """
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
        
        return h0, c0
    
    def get_model_info(self):
        """
        Return model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'LSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
        
        return info
    
    def print_model_summary(self):
        """
        Print model summary
        """
        info = self.get_model_info()
        print("=" * 50)
        print("LSTM MODEL SUMMARY")
        print("=" * 50)
        print(f"Type: {info['model_type']}")
        print(f"Input size: {info['input_size']}")
        print(f"Hidden size: {info['hidden_size']}")
        print(f"Number of layers: {info['num_layers']}")
        print(f"Output size: {info['output_size']}")
        print(f"Dropout rate: {info['dropout_rate']}")
        print(f"Bidirectional: {info['bidirectional']}")
        print(f"Uses attention: {info['use_attention']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        print("=" * 50)


class AttentionLayer(nn.Module):
    """
    Attention layer for LSTM model
    """
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output):
        """
        Apply attention to LSTM outputs
        """
        # lstm_output: (batch_size, seq_len, hidden_size)
        
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector


def create_lstm_variants():
    """
    Create different LSTM model variations for experimentation
    """
    variants = {
        'lstm_small': LSTMModel(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            output_size=1,
            dropout_rate=0.1,
            bidirectional=False,
            use_attention=False
        ),
        'lstm_medium': LSTMModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2,
            bidirectional=False,
            use_attention=False
        ),
        'lstm_large': LSTMModel(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=1,
            dropout_rate=0.3,
            bidirectional=False,
            use_attention=False
        ),
        'lstm_bidirectional': LSTMModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2,
            bidirectional=True,
            use_attention=False
        ),
        'lstm_attention': LSTMModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2,
            bidirectional=False,
            use_attention=True
        ),
        'lstm_bi_attention': LSTMModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2,
            bidirectional=True,
            use_attention=True
        )
    }
    
    return variants 