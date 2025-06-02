# 🧠 Work 2 - Deep Neural Networks
**Mackey-Glass Time Series Prediction with MLP, LSTM and GRU Architectures**

---

## 📋 Navigation

- [📖 README](README.md) ← **You are here**
- [🚀 How to Use](HOW_TO_USE.md)
- [📊 Executive Summary](EXECUTIVE_SUMMARY.md)
- [📈 Final Results](FINAL_RESULTS.md)
- [🌐 GitHub Pages Setup](GITHUB_PAGES_SETUP.md)

## 🌐 Online Demo

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://ratacheski.github.io/Mackey-Glass-Predicit/)

🔴 **Access the interactive report online**: [Click here to see the demo](https://ratacheski.github.io/Mackey-Glass-Predicit/report.html)

## 🎯 Objective

This project implements and compares **three neural network architectures** (MLP, LSTM, GRU) for Mackey-Glass time series prediction, exploring different configurations and advanced techniques for comprehensive performance analysis.

# Mackey-Glass Time Series Prediction with Neural Networks

This project implements three types of neural networks (MLP, LSTM, GRU) for Mackey-Glass time series prediction using PyTorch, with multiple configurations and variations for each model.

## 📚 Navigation

- **[📖 README.md](README.md)** - Complete documentation (you are here)
- **[🚀 HOW_TO_USE.md](HOW_TO_USE.md)** - Quick usage guide
- **[📊 EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Project overview
- **[📈 FINAL_RESULTS.md](FINAL_RESULTS.md)** - Experiment results

---

## Project Structure

```
mackey_glass_prediction/
├── config/
│   └── config.py                    # Centralized experiment configurations
├── data/
│   └── mackey_glass_generator.py    # Series generation and datasets
├── experiments/
│   └── run_experiment.py            # Main script to run experiments
├── models/
│   ├── __init__.py                  # Model imports
│   ├── mlp_model.py                 # MLP model with variations
│   ├── lstm_model.py                # LSTM model with variations
│   └── gru_model.py                 # GRU model with variations
├── utils/
│   ├── training.py                  # Training and evaluation functions
│   └── visualization/               # Complete visualization module
│       ├── __init__.py              # Basic plotting functions
│       ├── basic_plots.py           # Basic plots
│       ├── comparison_plots.py      # Comparison plots
│       ├── distribution_analysis.py # Distribution analysis
│       ├── interactive_html.py      # Interactive HTML reports
│       ├── reports.py               # Report generation
│       ├── statistical_tests.py    # Statistical tests
│       └── utils.py                 # Visualization utilities
├── generate_interactive_report.py   # Script to generate interactive reports
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mackey_glass_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

> **💡 For quick usage, see [HOW_TO_USE.md](HOW_TO_USE.md)**

### Run main experiments (three large models)

```bash
cd experiments
python run_experiment.py
```

By default, runs the main models: `mlp_large`, `lstm_large`, `gru_large`

### Run specific models

```bash
# Run only one model
python run_experiment.py --models mlp_medium

# Run multiple specific models
python run_experiment.py --models lstm_medium gru_medium mlp_small

# Run all available models
python run_experiment.py --models all
```

### Command line options

- `--models`: Specifies which models to run
  - `main`: Runs the three main models (default): `mlp_large`, `lstm_large`, `gru_large`
  - `all`: Runs all available configurations (15 models)
  - Specific names: any combination of available models
- `--no-save`: Don't save results (useful for quick tests)
- `--output-dir`: Custom prefix for output folder

### Programmatic usage example

```python
from experiments.run_experiment import run_single_experiment

# Run a single experiment
results = run_single_experiment('lstm_medium')

# Access metrics
print(f"RMSE: {results['metrics']['RMSE']:.6f}")
print(f"R²: {results['metrics']['R²']:.6f}")
print(f"EQMN1: {results['metrics']['EQMN1']:.6f}")
print(f"EQMN2: {results['metrics']['EQMN2']:.6f}")
```

## Available Models

### MLP (Multi-Layer Perceptron)
- **mlp_small**: 2 hidden layers [64, 32], ~3K parameters
- **mlp_medium**: 3 hidden layers [128, 64, 32], ~14K parameters  
- **mlp_large**: 4 hidden layers [256, 128, 64, 32], ~59K parameters

### LSTM (Long Short-Term Memory)
- **lstm_small**: 1 layer, 32 units, ~8K parameters
- **lstm_medium**: 2 layers, 64 units, ~50K parameters
- **lstm_large**: 3 layers, 128 units, ~200K parameters
- **lstm_bidirectional**: 2 bidirectional layers, 64 units
- **lstm_attention**: 2 layers with attention mechanism, 64 units

### GRU (Gated Recurrent Unit)
- **gru_small**: 1 layer, 32 units, ~6K parameters
- **gru_medium**: 2 layers, 64 units, ~37K parameters
- **gru_large**: 3 layers, 128 units, ~150K parameters
- **gru_bidirectional**: 2 bidirectional layers, 64 units
- **gru_attention**: 2 layers with attention mechanism, 64 units

## Configurations

Configurations are centralized in `config/config.py`:

### Mackey-Glass Series Parameters
```python
MACKEY_GLASS_CONFIG = {
    'n_points': 10000,  # Number of points
    'tau': 20,          # Delay parameter
    'gamma': 0.2,       # Gamma parameter
    'beta': 0.4,        # Beta parameter
    'n': 18,            # n parameter
    'x0': 0.8           # Initial value
}
```

### Dataset Configurations
```python
DATASET_CONFIG = {
    'window_size': 20,        # Input window size
    'prediction_steps': 1,    # Steps ahead to predict
    'train_ratio': 0.9,       # 90% for training, 10% for validation
    'batch_size': 8192,       # Batch size
    'shuffle_train': True     # Shuffle training data
}
```

### Training Configurations
```python
TRAINING_CONFIG = {
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 15,           # Early stopping
    'min_delta': 1e-6,        # Minimum improvement
    'use_scheduler': True,    # Learning rate scheduler
    'save_best_model': True,
    'save_final_model': True
}
```

## Results and Visualizations

After running experiments, the following are generated:

### 1. Output Structure
```
experiments/results/
├── final_report_[timestamp]/
│   ├── 01_overview_[timestamp].png          # Comparative overview
│   ├── [model]/                             # Folder for each model
│   │   ├── training_curves.png              # Training curves
│   │   ├── predictions.png                  # Predictions plot
│   │   ├── best_model.pth                   # Best saved model
│   │   └── final_model.pth                  # Final model
│   ├── 99_metrics_table_[timestamp].png     # Metrics table
│   ├── 99_metrics_comparison_[timestamp].png # Visual comparison
│   ├── metrics_table.csv                    # Metrics in CSV
│   └── report.html                          # Interactive HTML report
```

### 2. Interactive HTML Report
```bash
# Generate interactive demonstrative report
python generate_interactive_report.py
```

### 3. Evaluated Metrics

#### Basic Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

#### Normalized Metrics
- **EQMN1**: Mean Squared Error Normalized by Variance
  - **Formula**: EQMN1 = MSE / Var(y_true)
  - **Interpretation**: Values less than 0.1 are excellent, less than 0.5 are good
- **EQMN2**: Mean Squared Error Normalized by Naive Model
  - **Formula**: EQMN2 = MSE / MSE_naive (naive model)
  - **Interpretation**: < 1.0 indicates better than naive, < 0.5 is excellent

### 4. Generated Visualizations
- Training loss graphs
- Comparison between predictions and actual values
- Distribution analysis
- Q-Q and CDF graphs of residuals
- Visual comparison between models
- Formatted metric tables

## Mackey-Glass Series

The series is generated by the differential equation with delay:
```
dx/dt = βx(t-τ)/(1 + x(t-τ)^n) - γx(t)
```

Default parameters:
- τ = 20 (delay)
- β = 0.4
- γ = 0.2
- n = 18
- x₀ = 0.8 (initial condition)

## Development

### Adding new models

1. Create a new file in `models/` following the existing pattern
2. Implement the class inheriting from `torch.nn.Module`
3. Add required methods: `forward()`, `get_model_info()`, `print_model_summary()`
4. Update `models/__init__.py`
5. Add configuration in `config/config.py` in the `MODEL_CONFIGS` dictionary

### Example of new model
```python
# models/transformer_model.py
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super().__init__()
        # Model implementation
        
    def forward(self, x):
        # Forward pass
        
    def get_model_info(self):
        """Return model information for reports"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Transformer',
            # Add any specific model information
        }
        
    def print_model_summary(self):
        """Print model summary"""
        info = self.get_model_info()
        print(f"Model: {info['architecture']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
```

### Customizing Configurations

To create custom experiments, edit `config/config.py`:

```python
# Add new model configuration
MODEL_CONFIGS['my_custom_model'] = {
    'model_type': 'lstm',
    'input_size': 1,
    'hidden_size': 96,
    'num_layers': 4,
    'output_size': 1,
    'dropout_rate': 0.25,
    'bidirectional': True,
    'use_attention': True
}
```

## Primary Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computation
- **Matplotlib**: Basic visualization
- **Seaborn**: Statistical visualization
- **Pandas**: Data manipulation
- **Scikit-learn**: Metrics and utilities
- **SciPy**: Scientific computation
- **TQDM**: Progress bars

## Useful Links

- **[🚀 Quick Usage Guide](HOW_TO_USE.md)** - How to run experiments
- **[📊 Executive Summary](EXECUTIVE_SUMMARY.md)** - Project overview
- **[📈 Experiment Results](FINAL_RESULTS.md)** - Analysis of experimental results

## Contribution

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is under the MIT license. See LICENSE for more details.

## Author

**Rafael Ratacheski de Sousa Raulino**  
**MSc in Electrical Engineering and Computer Science - UFG**