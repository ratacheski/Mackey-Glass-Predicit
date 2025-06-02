# 📈 Final Results - Mackey-Glass Time Series Prediction

## 📚 Navigation

- **[📖 README.md](README.md)** - Complete documentation
- **[🚀 HOW_TO_USE.md](HOW_TO_USE.md)** - Quick usage guide
- **[📊 EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Project overview
- **[📈 FINAL_RESULTS.md](FINAL_RESULTS.md)** - Experiment results (you are here)

---

## 📊 Executive Summary of Experiments

This document presents the **final results** of Mackey-Glass time series prediction experiments using different neural network architectures. **7 models** were evaluated with optimized configurations.

**Experiment Date**: June 1, 2025  
**Number of Models Evaluated**: 7  
**Dataset**: 998 validation points  

## 🏆 General Model Ranking

### 🥇 1st Place: lstm_bidirectional
- **Total Score**: 5.5639
- **R²**: 0.990789 (**Exceptional**)
- **RMSE**: 0.036060 (**Best**)
- **MAE**: 0.026156
- **MAPE**: 4.98% (**Very low**)
- **EQMN1**: 0.009211 (**Excellent**)
- **EQMN2**: 0.159208 (**Much superior to naive**)

### 🥈 2nd Place: gru_bidirectional
- **Total Score**: 5.0892
- **R²**: 0.984880
- **RMSE**: 0.046200
- **MAE**: 0.035896
- **MAPE**: 7.06%
- **EQMN1**: 0.01512 (**Excellent**)
- **EQMN2**: 0.26133 (**Superior to naive**)

### 🥉 3rd Place: gru_large
- **Total Score**: 5.0521
- **R²**: 0.984210
- **RMSE**: 0.047213
- **MAE**: 0.036454
- **MAPE**: 8.66%
- **EQMN1**: 0.01579 (**Excellent**)
- **EQMN2**: 0.272916 (**Superior to naive**)

### 4th Place: lstm_large
- **Total Score**: 5.0431
- **R²**: 0.983583
- **RMSE**: 0.048141
- **MAE**: 0.035985
- **MAPE**: 6.56%
- **EQMN1**: 0.016417 (**Excellent**)
- **EQMN2**: 0.283757 (**Superior to naive**)

### 5th Place: gru_attention
- **Total Score**: 4.9242
- **R²**: 0.982544
- **RMSE**: 0.049641
- **MAE**: 0.039312
- **MAPE**: 8.25%
- **EQMN1**: 0.017456 (**Excellent**)
- **EQMN2**: 0.301715 (**Superior to naive**)

### 6th Place: lstm_attention
- **Total Score**: 4.8223
- **R²**: 0.980030
- **RMSE**: 0.053096
- **MAE**: 0.040231
- **MAPE**: 7.29%
- **EQMN1**: 0.01997 (**Good**)
- **EQMN2**: 0.345174 (**Superior to naive**)

### 7th Place: mlp_large
- **Total Score**: 2.7983
- **R²**: 0.932776
- **RMSE**: 0.097416
- **MAE**: 0.078291
- **MAPE**: 26.59% (**High**)
- **EQMN1**: 0.067224 (**Good**)
- **EQMN2**: 1.161922 (**Inferior to naive**)

## 📈 Detailed Metrics

### Complete Results Table

| Model | MSE | RMSE | MAE | MAPE (%) | R² | EQMN1 | EQMN2 | Classification |
|--------|-----|------|-----|----------|-------|-------|-------|---------------|
| **lstm_bidirectional** | 0.001300 | 0.036060 | 0.026156 | 4.98 | **0.990789** | **0.009211** | **0.159208** | 🥇 EXCEPTIONAL |
| **gru_bidirectional** | 0.002134 | 0.046200 | 0.035896 | 7.06 | 0.984880 | 0.01512 | 0.26133 | 🥈 EXCELLENT |
| **gru_large** | 0.002229 | 0.047213 | 0.036454 | 8.66 | 0.984210 | 0.01579 | 0.272916 | 🥉 EXCELLENT |
| **lstm_large** | 0.002318 | 0.048141 | 0.035985 | 6.56 | 0.983583 | 0.016417 | 0.283757 | 4th EXCELLENT |
| **gru_attention** | 0.002464 | 0.049641 | 0.039312 | 8.25 | 0.982544 | 0.017456 | 0.301715 | 5th EXCELLENT |
| **lstm_attention** | 0.002819 | 0.053096 | 0.040231 | 7.29 | 0.980030 | 0.01997 | 0.345174 | 6th EXCELLENT |
| **mlp_large** | 0.009490 | 0.097416 | 0.078291 | 26.59 | 0.932776 | 0.067224 | 1.161922 | 7th GOOD |

## 🔬 Technical Analysis of Results

### 📊 Performance by Architecture

#### LSTM (Long Short-Term Memory)
- **Best Model**: lstm_bidirectional (1st place)
- **Characteristics**: Bidirectional architectures outperformed unidirectional ones
- **Highlight**: lstm_bidirectional achieved the **best overall performance**
- **Limitation**: lstm_attention ranked 6th (attention mechanism was not effective)

#### GRU (Gated Recurrent Unit)
- **Best Model**: gru_bidirectional (2nd place)
- **Characteristics**: High consistency across all variations
- **Highlight**: Best **performance/complexity balance**
- **Observation**: gru_large (3rd) competed directly with lstm_large (4th)

#### MLP (Multi-Layer Perceptron)
- **Evaluated Model**: mlp_large (7th place)
- **Characteristics**: Inferior performance for time series
- **Limitation**: Only architecture with EQMN2 > 1.0 (worse than naive model)
- **Applicability**: Inadequate for complex time series predictions

### 🎯 Analysis of Normalized Metrics

#### EQMN1 (Normalized by Variance)
- **Excellent** (< 0.1): All recurrent models
- **Best**: lstm_bidirectional (0.009211)
- **Interpretation**: All RNN models captured data variability well

#### EQMN2 (Normalized by Naive Model)
- **Superior to Naive** (< 1.0): All except MLP
- **Best**: lstm_bidirectional (0.159208)
- **Critical**: mlp_large (1.161922) was **worse than naive model**

### 🧠 Residual Analysis

#### Models with Ideal Residuals
- **lstm_bidirectional**: No bias, symmetric
- **gru_bidirectional**: No bias, symmetric
- **gru_attention**: **Only normal residual distribution** (p > 0.05)

#### Models with Limitations
- **mlp_large**: Possible bias in residuals
- **lstm_large**: Asymmetry and excessive kurtosis
- **lstm_attention**: Normal kurtosis, but not normality

## 💡 Recommendations by Use Case

### 🎯 For Maximum Precision
**Recommended**: **lstm_bidirectional**
- **R² = 0.990789** (best)
- **RMSE = 0.036060** (lowest error)
- **MAPE = 4.98%** (very low percentage error)
- **Application**: Critical systems, scientific research

### ⚖️ For Performance/Efficiency Balance
**Recommended**: **gru_bidirectional**
- **R² = 0.984880** (excellent)
- **Lower complexity** than equivalent LSTM
- **Well-behaved residuals**
- **Application**: Production systems, commercial applications

### 🚀 For Computationally Constrained Systems
**Recommended**: **gru_large**
- **R² = 0.984210** (excellent)
- **Unidirectional architecture** (lower computational cost)
- **Good generalization**
- **Application**: Embedded devices, real-time processing

### ❌ Not Recommended
**mlp_large**: Inadequate performance for time series
- **EQMN2 > 1.0**: Worse than naive model
- **MAPE = 26.59%**: Very high error
- **Application**: Only for educational/comparison purposes

## 📊 Important Insights

### 1. **Superiority of Bidirectional Architectures**
- **lstm_bidirectional** and **gru_bidirectional** occupied the **top two places**
- Ability to process past and future information is crucial for time series

### 2. **Limited Effectiveness of Attention Mechanism**
- **lstm_attention** (6th) and **gru_attention** (5th) had inferior performance to bidirectional versions
- For simple time series, attention may introduce unnecessary complexity

### 3. **MLP Inadequacy for Time Series**
- **mlp_large** was the only model with **EQMN2 > 1.0**
- Confirms importance of temporal memory in recurrent networks

### 4. **Consistency of Normalized Metrics**
- **EQMN1 < 0.02** for all RNN models (excellent)
- **EQMN2 < 0.35** for all RNN models (much superior to naive)

## 🎯 Experimental Configuration

### Mackey-Glass Series Parameters
- **Points**: 10,000
- **Parameters**: τ=20, β=0.4, γ=0.2, n=18, x₀=0.8
- **Input window**: 20 points
- **Prediction**: 1 step ahead

### Training Configurations
- **Maximum Epochs**: 150
- **Early stopping**: 15 epochs of patience
- **Learning rate**: 1e-3 with scheduler
- **Optimizer**: Adam
- **Regularization**: L2 (1e-5) + Dropout

### Dataset
- **Split**: 90% train, 10% validation
- **Validation points**: 998
- **Batch size**: 8192
- **Normalization**: Min-Max Scaling

## 📁 Generated Files

### Results Structure
```
results/final_report_1748824486/
├── relatorio_textual_20250601_213446.txt      # Detailed report
├── 99_tabela_metricas_20250601_213446.csv     # Metrics in CSV
├── 01_visao_geral_20250601_213446.png         # Visual comparison
├── 99_tabela_metricas_20250601_213446.png     # Formatted table
├── 99_comparacao_metricas_20250601_213446.png # Comparison graphics
├── [modelo]/                                  # Folder for each model:
│   ├── training_curves.png                    # Training curves
│   ├── predictions.png                        # Predictions vs real
│   ├── [modelo]_qq_plot_[timestamp].png       # Q-Q analysis
│   ├── [modelo]_cdf_[timestamp].png           # CDF analysis
│   ├── best_model.pth                         # Best model
│   └── final_model.pth                        # Final model
└── report.html                             # Interactive report
```

## 🔍 Next Steps

### For Research
1. **Investigate hybrid architectures**: Combine LSTM/GRU with temporal attention
2. **Test transformers**: Evaluate attention-based architectures
3. **Interpretability analysis**: Understand what models learned

### For Production
1. **Inference optimization**: Quantization and pruning of lstm_bidirectional
2. **Drift monitoring**: Detect changes in data distribution
3. **Ensemble methods**: Combine predictions from best models

### For Validation
1. **Test on longer series**: Evaluate temporal generalization
2. **Temporal cross-validation**: Validation with multiple temporal windows
3. **Robustness test**: Evaluate with different levels of noise

## 🚀 How to Reproduce Results

### Minimal Configuration
```bash
cd mackey_glass_prediction/experiments
python run_experiment.py --models lstm_bidirectional gru_bidirectional gru_large
```

### Complete Configuration
```bash
# Reproduce all results
python run_experiment.py --models mlp_large lstm_large lstm_bidirectional lstm_attention gru_large gru_bidirectional gru_attention

# Generate interactive report
cd ..
python generate_interactive_report.py
```

## 📚 Useful Links

- **[📖 README.md](README.md)** - Complete technical documentation
- **[🚀 HOW_TO_USE.md](HOW_TO_USE.md)** - Practical execution guide  
- **[📊 EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Strategic project overview

---

## �� Final Conclusions

### Main Discoveries
1. **LSTM Bidirectional is the superior model** for Mackey-Glass prediction
2. **Bidirectional architectures outperform unidirectional ones** consistently
3. **GRU offers excellent cost-benefit** compared to LSTM
4. **MLP is inadequate** for series with long dependencies
5. **Attention mechanisms** were not effective in this specific context

### Practical Implications
- For **critical applications**: Use lstm_bidirectional
- For **general production**: Use gru_bidirectional or gru_large
- For **research**: Explore hybrid and transformer architectures
- **Avoid MLP** for complex time series predictions

### Contributions from the Work
- **Comprehensive comparison** of architectures in chaotic time series
- **Detailed analysis** with normalized metrics (EQMN1, EQMN2)
- **Reproducible methodology** with well-documented code
- **Practical insights** for model selection in production

---

**Experiment conducted on**: June 1, 2025  
**Author**: Rafael Ratacheski de Sousa Raulino  
**Course**: Deep Neural Networks - UFG 2025/1  
**Status**: ✅ **COMPLETE ANALYSIS**