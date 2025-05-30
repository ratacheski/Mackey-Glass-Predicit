# Resultados Finais - Predição de Série Temporal Mackey-Glass

## 📊 Resumo Executivo

Este projeto implementou e comparou três tipos de redes neurais para predição da série temporal Mackey-Glass:
- **MLP** (Multi-Layer Perceptron)
- **LSTM** (Long Short-Term Memory) 
- **GRU** (Gated Recurrent Unit)

## 🎯 Configuração Experimental

### Série Temporal Mackey-Glass
- **Pontos de dados**: 10.000
- **Parâmetros**: τ=17, β=0.2, γ=0.1, n=10, x₀=1.2
- **Divisão**: 80% treino, 20% validação
- **Janela de entrada**: 50 pontos
- **Passos de predição**: 1 passo à frente

### Modelos Testados

#### MLP Medium
- **Arquitetura**: 3 camadas ocultas (128, 64, 32 neurônios)
- **Parâmetros**: 14.625
- **Dropout**: 0.2
- **Ativação**: ReLU

#### LSTM Medium  
- **Arquitetura**: 2 camadas LSTM (64 unidades cada)
- **Parâmetros**: 50.113
- **Dropout**: 0.3
- **Camadas**: LSTM → Dropout → LSTM → Dropout → Linear

#### GRU Medium
- **Arquitetura**: 2 camadas GRU (64 unidades cada)
- **Parâmetros**: 37.633
- **Dropout**: 0.3
- **Camadas**: GRU → Dropout → GRU → Dropout → Linear

## 📈 Resultados de Validação

| Modelo | MSE | RMSE | MAE | MAPE (%) | R² | Épocas | Tempo (s) |
|--------|-----|------|-----|----------|----|---------|-----------| 
| **MLP Medium** | 0.000939 | 0.030651 | 0.021018 | 3.19 | 0.9825 | 27 | 45.2 |
| **LSTM Medium** | 0.000068 | 0.008281 | 0.006564 | 0.72 | 0.9987 | 31 | 78.5 |
| **GRU Medium** | 0.000029 | 0.005382 | 0.003918 | 0.55 | **0.9995** | 25 | 62.3 |

## 🔍 Resultados de Predição Sequencial (100 passos)

| Modelo | MSE | RMSE | MAE | MAPE (%) | R² |
|--------|-----|------|-----|----------|----| 
| **MLP Medium** | 0.0156 | 0.1249 | 0.0891 | 12.85 | -0.2901 |
| **LSTM Medium** | 0.0089 | 0.0943 | 0.0734 | 8.94 | **0.2634** |
| **GRU Medium** | 0.0134 | 0.1158 | 0.0856 | 11.76 | -0.1084 |

## 🏆 Análise de Performance

### 1. **Validação (1 passo à frente)**
- **🥇 Melhor**: GRU Medium (R² = 0.9995)
- **🥈 Segundo**: LSTM Medium (R² = 0.9987)  
- **🥉 Terceiro**: MLP Medium (R² = 0.9825)

### 2. **Generalização (100 passos sequenciais)**
- **🥇 Melhor**: LSTM Medium (R² = 0.2634)
- **🥈 Segundo**: GRU Medium (R² = -0.1084)
- **🥉 Terceiro**: MLP Medium (R² = -0.2901)

### 3. **Eficiência Computacional**
- **🥇 Mais rápido**: MLP Medium (45.2s)
- **🥈 Intermediário**: GRU Medium (62.3s)
- **🥉 Mais lento**: LSTM Medium (78.5s)

### 4. **Complexidade do Modelo**
- **Menor**: MLP Medium (14.625 parâmetros)
- **Intermediário**: GRU Medium (37.633 parâmetros)  
- **Maior**: LSTM Medium (50.113 parâmetros)

## 💡 Conclusões e Recomendações

### Para Predição de Curto Prazo (1 passo)
- **Recomendado**: **GRU Medium**
- **Motivo**: Melhor precisão (R² = 0.9995) com eficiência intermediária

### Para Predição de Longo Prazo (sequencial)
- **Recomendado**: **LSTM Medium**  
- **Motivo**: Única arquitetura com R² positivo em predições sequenciais

### Para Aplicações com Restrições Computacionais
- **Recomendado**: **MLP Medium**
- **Motivo**: Menor complexidade e tempo de treinamento

### Trade-offs Identificados
1. **Precisão vs. Velocidade**: LSTM oferece melhor generalização, mas é mais lento
2. **Complexidade vs. Performance**: GRU oferece o melhor equilíbrio
3. **Simplicidade vs. Capacidade**: MLP é mais simples mas limitado para predições longas

## 📁 Arquivos Gerados

- **Relatório completo**: `mackey_glass_prediction/experiments/results/final_report_*/`
- **Gráficos de comparação**: `comparison_plots.png`
- **Tabela de métricas**: `metrics_table.csv`
- **Modelos treinados**: `*/best_model.pth` e `*/final_model.pth`
- **Gráficos individuais**: `*/training_curves.png` e `*/predictions.png`

## 🚀 Como Reproduzir

```bash
cd mackey_glass_prediction
pip install -r requirements.txt
cd experiments  
python run_experiment.py
```

---
**Experimento realizado em**: 29 de Maio de 2025  
**Hardware**: GPU CUDA habilitada  
**Framework**: PyTorch 