import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    Modelo MLP (Multi-Layer Perceptron) para predição de séries temporais
    """
    
    def __init__(self, input_size=20, hidden_sizes=[128, 64, 32], output_size=1, 
                 dropout_rate=0.2, activation='relu'):
        """
        Parâmetros:
        - input_size: tamanho da janela de entrada
        - hidden_sizes: lista com o número de neurônios em cada camada oculta
        - output_size: número de passos à frente para predizer
        - dropout_rate: taxa de dropout para regularização
        - activation: função de ativação ('relu', 'tanh', 'leaky_relu')
        """
        super(MLPModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Criar camadas da rede
        layers = []
        
        # Primeira camada
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(self._get_activation())
        layers.append(nn.Dropout(dropout_rate))
        
        # Camadas ocultas
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(dropout_rate))
        
        # Camada de saída
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _get_activation(self):
        """
        Retorna a função de ativação escolhida
        """
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()  # padrão
    
    def _initialize_weights(self):
        """
        Inicializa os pesos da rede
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass da rede
        """
        # Flatten da entrada se necessário
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def get_model_info(self):
        """
        Retorna informações sobre o modelo
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
        Imprime um resumo do modelo
        """
        info = self.get_model_info()
        print("=" * 50)
        print("RESUMO DO MODELO MLP")
        print("=" * 50)
        print(f"Tipo: {info['model_type']}")
        print(f"Tamanho da entrada: {info['input_size']}")
        print(f"Camadas ocultas: {info['hidden_sizes']}")
        print(f"Tamanho da saída: {info['output_size']}")
        print(f"Taxa de dropout: {info['dropout_rate']}")
        print(f"Função de ativação: {info['activation']}")
        print(f"Total de parâmetros: {info['total_parameters']:,}")
        print(f"Parâmetros treináveis: {info['trainable_parameters']:,}")
        print("=" * 50)


def create_mlp_variants():
    """
    Cria diferentes variações do modelo MLP para experimentação
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