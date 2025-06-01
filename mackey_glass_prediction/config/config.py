import torch

# Configurações gerais
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

# Configurações da série temporal de Mackey-Glass
MACKEY_GLASS_CONFIG = {
    'n_points': 10000,  # Número de pontos a gerar
    'tau': 20,         # Delay parameter
    'gamma': 0.2,      # Gamma parameter
    'beta': 0.4,       # Beta parameter
    'n': 18,           # N parameter
    'x0': 0.8          # Initial value
}

# Configurações do dataset
DATASET_CONFIG = {
    'window_size': 20,        # Tamanho da janela de entrada
    'prediction_steps': 1,    # Número de passos à frente para predizer
    'train_ratio': 0.9,       # Proporção de dados para treino (90%)
    'batch_size': 8192,         # Tamanho do batch
    'shuffle_train': True,    # Embaralhar dados de treino
    'num_workers': 0          # Número de workers para DataLoader
}

# Configurações de treinamento
TRAINING_CONFIG = {
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 15,           # Early stopping
    'min_delta': 1e-6,        # Mínima melhoria para considerar progresso
    'use_scheduler': True,    # Usar scheduler de learning rate
    'save_best_model': True,
    'save_final_model': True
}

# Configurações específicas para cada modelo
MODEL_CONFIGS = {
    'mlp_small': {
        'model_type': 'mlp',
        'input_size': DATASET_CONFIG['window_size'],
        'hidden_sizes': [64, 32],
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.1,
        'activation': 'relu'
    },
    'mlp_medium': {
        'model_type': 'mlp',
        'input_size': DATASET_CONFIG['window_size'],
        'hidden_sizes': [128, 64, 32],
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.2,
        'activation': 'relu'
    },
    'mlp_large': {
        'model_type': 'mlp',
        'input_size': DATASET_CONFIG['window_size'],
        'hidden_sizes': [256, 128, 64, 32],
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.3,
        'activation': 'relu'
    },
    'lstm_small': {
        'model_type': 'lstm',
        'input_size': 1,
        'hidden_size': 32,
        'num_layers': 1,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.1,
        'bidirectional': False,
        'use_attention': False
    },
    'lstm_medium': {
        'model_type': 'lstm',
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.2,
        'bidirectional': False,
        'use_attention': False
    },
    'lstm_large': {
        'model_type': 'lstm',
        'input_size': 1,
        'hidden_size': 128,
        'num_layers': 3,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.3,
        'bidirectional': False,
        'use_attention': False
    },
    'lstm_bidirectional': {
        'model_type': 'lstm',
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.2,
        'bidirectional': True,
        'use_attention': False
    },
    'lstm_attention': {
        'model_type': 'lstm',
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.2,
        'bidirectional': False,
        'use_attention': True
    },
    'gru_small': {
        'model_type': 'gru',
        'input_size': 1,
        'hidden_size': 32,
        'num_layers': 1,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.1,
        'bidirectional': False,
        'use_attention': False
    },
    'gru_medium': {
        'model_type': 'gru',
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.2,
        'bidirectional': False,
        'use_attention': False
    },
    'gru_large': {
        'model_type': 'gru',
        'input_size': 1,
        'hidden_size': 128,
        'num_layers': 3,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.3,
        'bidirectional': False,
        'use_attention': False
    },
    'gru_bidirectional': {
        'model_type': 'gru',
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.2,
        'bidirectional': True,
        'use_attention': False
    },
    'gru_attention': {
        'model_type': 'gru',
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': DATASET_CONFIG['prediction_steps'],
        'dropout_rate': 0.2,
        'bidirectional': False,
        'use_attention': True
    }
}

# Configurações de saída
OUTPUT_CONFIG = {
    'base_output_dir': './results',
    'save_plots': True,
    'save_models': True,
    'save_data': True,
    'plot_format': 'png',
    'plot_dpi': 300
}

# Configurações de visualização
VISUALIZATION_CONFIG = {
    'n_points_plot': 500,     # Número máximo de pontos para visualização
    'figsize_small': (10, 6),
    'figsize_medium': (12, 8),
    'figsize_large': (15, 10),
    'style': 'ggplot'
}

# Função para obter configuração completa para um experimento
def get_experiment_config(model_name):
    """
    Retorna configuração completa para um experimento específico
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Modelo '{model_name}' não encontrado na configuração")
    
    config = {
        'model': MODEL_CONFIGS[model_name],
        'dataset': DATASET_CONFIG,
        'training': TRAINING_CONFIG,
        'mackey_glass': MACKEY_GLASS_CONFIG,
        'output': OUTPUT_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'device': DEVICE,
        'random_seed': RANDOM_SEED
    }
    
    # Adicionar caminhos específicos do modelo
    model_output_dir = f"{OUTPUT_CONFIG['base_output_dir']}/{model_name}"
    config['training']['model_save_path'] = f"{model_output_dir}/best_model.pth"
    config['training']['final_model_save_path'] = f"{model_output_dir}/final_model.pth"
    
    return config

# Lista dos modelos principais (um de cada tipo)
MAIN_MODELS = ['mlp_large', 'lstm_large', 'gru_large']

# Lista de todos os modelos disponíveis
ALL_MODELS = list(MODEL_CONFIGS.keys()) 