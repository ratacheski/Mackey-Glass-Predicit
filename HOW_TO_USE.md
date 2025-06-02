# How to Use the Project - Quick Guide

## 📚 Navigation

- **[📖 README.md](README.md)** - Complete documentation
- **[🚀 HOW_TO_USE.md](HOW_TO_USE.md)** - Quick usage guide (you are here)
- **[📊 EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Project overview
- **[📈 FINAL_RESULTS.md](FINAL_RESULTS.md)** - Experiment results

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd mackey_glass_prediction
pip install -r requirements.txt
```

### 2. Run Experiments

#### Main models (recommended for comparison)
```bash
cd experiments
python run_experiment.py
```
**Runs**: `mlp_large`, `lstm_large`, `gru_large` (one model of each type)

#### Specific model
```bash
# Only a medium model
python run_experiment.py --models mlp_medium

# Only medium LSTM
python run_experiment.py --models lstm_medium

# Only medium GRU
python run_experiment.py --models gru_medium

# Multiple specific models
python run_experiment.py --models mlp_medium lstm_medium gru_small
```

#### All available configurations
```bash
python run_experiment.py --models all
```
**Runs**: All 15 available models (3 sizes + 2 variations for each type)

#### Quick test (without saving)
```bash
python run_experiment.py --models mlp_small --no-save
```

#### Custom output folder
```bash
python run_experiment.py --output-dir my_experiment
```

## 📊 Available Models

> **📖 For complete details, see [README.md](README.md#available-models)**

### MLP (Multi-Layer Perceptron)
- `mlp_small`: 2 layers [64, 32], ~3K parameters
- `mlp_medium`: 3 layers [128, 64, 32], ~14K parameters  
- `mlp_large`: 4 layers [256, 128, 64, 32], ~59K parameters

### LSTM (Long Short-Term Memory)
- `lstm_small`: 1 layer, 32 units, ~8K parameters
- `lstm_medium`: 2 layers, 64 units, ~50K parameters
- `lstm_large`: 3 layers, 128 units, ~200K parameters
- `lstm_bidirectional`: 2 bidirectional layers, 64 units
- `lstm_attention`: 2 layers with attention, 64 units

### GRU (Gated Recurrent Unit)
- `gru_small`: 1 layer, 32 units, ~6K parameters
- `gru_medium`: 2 layers, 64 units, ~37K parameters
- `gru_large`: 3 layers, 128 units, ~150K parameters
- `gru_bidirectional`: 2 bidirectional layers, 64 units
- `gru_attention`: 2 layers with attention, 64 units

## 🔧 Default Configurations

### Mackey-Glass Series
- **Points**: 10,000
- **Input window**: 20 points
- **Prediction**: 1 step ahead
- **Split**: 90% training, 10% validation

### Training
- **Maximum epochs**: 150
- **Early stopping**: 15 epochs patience
- **Learning rate**: 1e-3 with scheduler
- **Batch size**: 8192

## 📁 Results Structure

```
experiments/results/
├── final_report_20250529_143052/         # Folder with timestamp
│   ├── 01_overview_20250529_143052.png       # Overview
│   ├── mlp_large/                        # Results by model
│   │   ├── training_curves.png           # Training curves
│   │   ├── predictions.png               # Predictions plot
│   │   ├── best_model.pth               # Best saved model
│   │   └── final_model.pth              # Final model
│   ├── 99_metrics_table_20250529_143052.png
│   ├── 99_metrics_comparison_20250529_143052.png
│   ├── metrics_table.csv                # Metrics in CSV
│   └── report.html                   # Interactive report
```

## 📈 Reported Metrics

### Basic Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Normalized Metrics
- **EQMN1**: Mean Squared Error Normalized by Variance
  - Formula: MSE / Var(y_true)
  - Interpretation: < 0.1 excellent, < 0.5 good
- **EQMN2**: Mean Squared Error Normalized by Naive Model
  - Formula: MSE / MSE_naive
  - Interpretation: < 1.0 better than naive, < 0.5 excellent

## 🎨 Interactive HTML Report

Generate an interactive demonstration report:
```bash
python generate_interactive_report.py
```

**Report features**:
- 📊 Detailed metrics (R², RMSE, MAE, MSE, MAPE, EQMN1, EQMN2)
- 🖼️ Full-screen image visualization
- 📈 Charts organized by model
- 📋 Interactive comparisons
- 👨‍🎓 Author information

## ⚙️ Customization

> **📖 For advanced configurations, see [README.md](README.md#configurations)**

### Modify Series Configurations
Edit `config/config.py`:
```python
MACKEY_GLASS_CONFIG = {
    'n_points': 5000,    # Fewer points for quick test
    'tau': 20,
    'gamma': 0.2,
    'beta': 0.4,
    'n': 18,
    'x0': 0.8
}
```

### Modify Training Parameters
```python
TRAINING_CONFIG = {
    'epochs': 50,        # Fewer epochs for testing
    'learning_rate': 5e-4,
    'patience': 10,      # More aggressive early stopping
    # ...
}
```

### Modify Dataset
```python
DATASET_CONFIG = {
    'window_size': 10,   # Smaller window
    'batch_size': 4096,  # Smaller batch if low memory
    'train_ratio': 0.8,  # More data for validation
    # ...
}
```

## 🔍 Troubleshooting

### CUDA Error
The code automatically detects GPU/CPU. If there are problems:
```python
# In config/config.py
DEVICE = 'cpu'  # Force CPU usage
```

### Memory Error
Reduce batch size:
```python
DATASET_CONFIG = {
    'batch_size': 1024,  # Reduce from 8192
    # ...
}
```

### Missing Dependencies
```bash
pip install torch numpy matplotlib seaborn tqdm scipy scikit-learn pandas
```

### Erro de Path/Import
Certifique-se de estar no diretório correto:
```bash
cd mackey_glass_prediction/experiments
python run_experiment.py
```

## 💡 Usage Tips

### For Quick Comparison
```bash
# Small models for testing
python run_experiment.py --models mlp_small lstm_small gru_small --no-save
```

### For Complete Experiment
```bash
# All models (may take several hours)
python run_experiment.py --models all --output-dir complete_experiment
```

### For Specific Model with Analysis
```bash
# One model with saved results
python run_experiment.py --models lstm_medium
```

### For Reproducing Results
The code uses fixed seeds (RANDOM_SEED = 42) for reproducibility. 
To get different results, modify in `config/config.py`:
```python
RANDOM_SEED = 123  # Or any other value
```

## 🎯 Common Scenarios

### Development/Debug
```bash
python run_experiment.py --models mlp_small --no-save
```

### Scientific Comparison
```bash
python run_experiment.py  # Main models
```

### Complete Analysis
```bash
python run_experiment.py --models all
```

### Model for Production
```bash
python run_experiment.py --models gru_medium  # Good balance
```

## 📊 Results Interpretation

### CSV File
```csv
Model,MSE,RMSE,MAE,MAPE,R²,EQMN1,EQMN2,Training_Time,Parameters
mlp_large,0.000234,0.015301,0.011234,1.89,0.9912,0.0456,0.2341,67.3,59073
```

### Console Output
```
LSTM_MEDIUM completed successfully!
Training time: 78.5 seconds
Best validation loss: 0.000068
Epochs trained: 31
Parameters: 50,113

VALIDATION METRICS:
MSE: 0.000068
RMSE: 0.008281
MAE: 0.006564
MAPE: 0.72%
R²: 0.9987
EQMN1: 0.0425
EQMN2: 0.1234
```

## 📚 Useful Links

- **[📖 README.md](README.md)** - Complete documentation and advanced configurations
- **[📊 EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Project strategic overview
- **[📈 FINAL_RESULTS.md](FINAL_RESULTS.md)** - Experiment results analysis

---
**Developed by**: Rafael Ratacheski de Sousa Raulino  
**Course**: Deep Neural Networks - UFG 2025/1