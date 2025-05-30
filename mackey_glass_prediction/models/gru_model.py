import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    """
    Modelo GRU (Gated Recurrent Unit) para predição de séries temporais
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1,
                 dropout_rate=0.2, bidirectional=False, use_attention=False):
        """
        Parâmetros:
        - input_size: número de features de entrada (1 para série univariada)
        - hidden_size: número de unidades GRU em cada camada
        - num_layers: número de camadas GRU empilhadas
        - output_size: número de passos à frente para predizer
        - dropout_rate: taxa de dropout para regularização
        - bidirectional: se True, usa GRU bidirecional
        - use_attention: se True, adiciona mecanismo de atenção
        """
        super(GRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Camada GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calcular o tamanho da saída do GRU
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Camada de atenção (opcional)
        if use_attention:
            self.attention = AttentionLayer(gru_output_size)
        
        # Camadas fully connected
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicializa os pesos da rede
        """
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass da rede
        """
        batch_size = x.size(0)
        
        # Reshape para (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Adicionar dimensão de feature
        
        # Inicializar estado oculto
        h0 = self._init_hidden(batch_size, x.device)
        
        # Forward pass através do GRU
        gru_out, hidden = self.gru(x, h0)
        
        # Aplicar atenção se configurado
        if self.use_attention:
            gru_out = self.attention(gru_out)
        else:
            # Usar apenas a última saída da sequência
            gru_out = gru_out[:, -1, :]
        
        # Forward pass através das camadas fully connected
        output = self.fc_layers(gru_out)
        
        return output
    
    def _init_hidden(self, batch_size, device):
        """
        Inicializa o estado oculto do GRU
        """
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
        
        return h0
    
    def get_model_info(self):
        """
        Retorna informações sobre o modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'GRU',
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
        Imprime um resumo do modelo
        """
        info = self.get_model_info()
        print("=" * 50)
        print("RESUMO DO MODELO GRU")
        print("=" * 50)
        print(f"Tipo: {info['model_type']}")
        print(f"Tamanho da entrada: {info['input_size']}")
        print(f"Tamanho oculto: {info['hidden_size']}")
        print(f"Número de camadas: {info['num_layers']}")
        print(f"Tamanho da saída: {info['output_size']}")
        print(f"Taxa de dropout: {info['dropout_rate']}")
        print(f"Bidirecional: {info['bidirectional']}")
        print(f"Usa atenção: {info['use_attention']}")
        print(f"Total de parâmetros: {info['total_parameters']:,}")
        print(f"Parâmetros treináveis: {info['trainable_parameters']:,}")
        print("=" * 50)


class AttentionLayer(nn.Module):
    """
    Camada de atenção para o modelo GRU
    """
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, gru_output):
        """
        Aplica atenção às saídas do GRU
        """
        # gru_output: (batch_size, seq_len, hidden_size)
        
        # Calcular pesos de atenção
        attention_weights = torch.softmax(self.attention(gru_output), dim=1)
        
        # Aplicar pesos de atenção
        context_vector = torch.sum(attention_weights * gru_output, dim=1)
        
        return context_vector


def create_gru_variants():
    """
    Cria diferentes variações do modelo GRU para experimentação
    """
    variants = {
        'gru_small': GRUModel(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            output_size=1,
            dropout_rate=0.1,
            bidirectional=False,
            use_attention=False
        ),
        'gru_medium': GRUModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2,
            bidirectional=False,
            use_attention=False
        ),
        'gru_large': GRUModel(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=1,
            dropout_rate=0.3,
            bidirectional=False,
            use_attention=False
        ),
        'gru_bidirectional': GRUModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2,
            bidirectional=True,
            use_attention=False
        ),
        'gru_attention': GRUModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2,
            bidirectional=False,
            use_attention=True
        ),
        'gru_bi_attention': GRUModel(
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