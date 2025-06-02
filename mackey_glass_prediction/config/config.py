import torch

# General configurations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

# Mackey-Glass time series configurations
MACKEY_GLASS_CONFIG = {
    'n_points': 10000,  # Number of points to generate
    'tau': 20,         # Delay parameter
    'gamma': 0.2,      # Gamma parameter
    'beta': 0.4,       # Beta parameter
    'n': 18,           # N parameter
    'x0': 0.8          # Initial value
}

# Dataset configurations
DATASET_CONFIG = {
    'window_size': 20,        # Input window size
    'prediction_steps': 1,    # Number of steps ahead to predict
    'train_ratio': 0.9,       # Proportion of data for training (90%)
    'batch_size': 8192,         # Batch size
    'shuffle_train': True,    # Shuffle training data
    'num_workers': 0          # Number of workers for DataLoader
}

# Training configurations
TRAINING_CONFIG = {
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 15,           # Early stopping
    'min_delta': 1e-6,        # Minimum improvement to consider progress
    'use_scheduler': True,    # Use learning rate scheduler
    'save_best_model': True,
    'save_final_model': True
}

# Specific configurations for each model
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

# Output configurations
OUTPUT_CONFIG = {
    'base_output_dir': './results',
    'save_plots': True,
    'save_models': True,
    'save_data': True,
    'plot_format': 'png',
    'plot_dpi': 300
}

# Visualization configurations
VISUALIZATION_CONFIG = {
    'n_points_plot': 500,     # Maximum number of points for visualization
    'figsize_small': (10, 6),
    'figsize_medium': (12, 8),
    'figsize_large': (15, 10),
    'style': 'ggplot'
}

# Function to get complete configuration for an experiment
def get_experiment_config(model_name):
    """
    Returns complete configuration for a specific experiment
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found in configuration")
    
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
    
    # Add model-specific paths
    model_output_dir = f"{OUTPUT_CONFIG['base_output_dir']}/{model_name}"
    config['training']['model_save_path'] = f"{model_output_dir}/best_model.pth"
    config['training']['final_model_save_path'] = f"{model_output_dir}/final_model.pth"
    
    return config

# List of main models (one of each type)
MAIN_MODELS = ['mlp_large', 'lstm_large', 'gru_large']

# List of all available models
ALL_MODELS = list(MODEL_CONFIGS.keys()) 