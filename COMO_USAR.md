# Como Usar o Projeto - Guia Rápido

## 📚 Navegação

- **[📖 README.md](README.md)** - Documentação completa
- **[🚀 COMO_USAR.md](COMO_USAR.md)** - Guia rápido de uso (você está aqui)
- **[📊 RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md)** - Visão geral do projeto
- **[📈 RESULTADOS_FINAIS.md](RESULTADOS_FINAIS.md)** - Resultados dos experimentos

---

## 🚀 Início Rápido

### 1. Instalar Dependências
```bash
cd mackey_glass_prediction
pip install -r requirements.txt
```

### 2. Executar Experimentos

#### Modelos principais (recomendado para comparação)
```bash
cd experiments
python run_experiment.py
```
**Executa**: `mlp_large`, `lstm_large`, `gru_large` (um modelo de cada tipo)

#### Modelo específico
```bash
# Apenas um modelo médio
python run_experiment.py --models mlp_medium

# Apenas LSTM médio
python run_experiment.py --models lstm_medium

# Apenas GRU médio
python run_experiment.py --models gru_medium

# Múltiplos modelos específicos
python run_experiment.py --models mlp_medium lstm_medium gru_small
```

#### Todas as configurações disponíveis
```bash
python run_experiment.py --models all
```
**Executa**: Todos os 15 modelos disponíveis (3 tamanhos + 2 variações para cada tipo)

#### Teste rápido (sem salvar)
```bash
python run_experiment.py --models mlp_small --no-save
```

#### Pasta de saída personalizada
```bash
python run_experiment.py --output-dir meu_experimento
```

## 📊 Modelos Disponíveis

> **📖 Para detalhes completos, consulte [README.md](README.md#modelos-disponíveis)**

### MLP (Multi-Layer Perceptron)
- `mlp_small`: 2 camadas [64, 32], ~3K parâmetros
- `mlp_medium`: 3 camadas [128, 64, 32], ~14K parâmetros  
- `mlp_large`: 4 camadas [256, 128, 64, 32], ~59K parâmetros

### LSTM (Long Short-Term Memory)
- `lstm_small`: 1 camada, 32 unidades, ~8K parâmetros
- `lstm_medium`: 2 camadas, 64 unidades, ~50K parâmetros
- `lstm_large`: 3 camadas, 128 unidades, ~200K parâmetros
- `lstm_bidirectional`: 2 camadas bidirecionais, 64 unidades
- `lstm_attention`: 2 camadas com atenção, 64 unidades

### GRU (Gated Recurrent Unit)
- `gru_small`: 1 camada, 32 unidades, ~6K parâmetros
- `gru_medium`: 2 camadas, 64 unidades, ~37K parâmetros
- `gru_large`: 3 camadas, 128 unidades, ~150K parâmetros
- `gru_bidirectional`: 2 camadas bidirecionais, 64 unidades
- `gru_attention`: 2 camadas com atenção, 64 unidades

## 🔧 Configurações Padrão

### Série Mackey-Glass
- **Pontos**: 10.000
- **Janela de entrada**: 20 pontos
- **Predição**: 1 passo à frente
- **Divisão**: 90% treino, 10% validação

### Treinamento
- **Épocas máximas**: 150
- **Early stopping**: 15 épocas de paciência
- **Learning rate**: 1e-3 com scheduler
- **Batch size**: 8192

## 📁 Estrutura dos Resultados

```
experiments/results/
├── final_report_20250529_143052/         # Pasta com timestamp
│   ├── 01_visao_geral_20250529_143052.png    # Visão geral
│   ├── mlp_large/                        # Resultados por modelo
│   │   ├── training_curves.png           # Curvas de treinamento
│   │   ├── predictions.png               # Gráfico de predições
│   │   ├── best_model.pth               # Melhor modelo salvo
│   │   └── final_model.pth              # Modelo final
│   ├── 99_tabela_metricas_20250529_143052.png
│   ├── 99_comparacao_metricas_20250529_143052.png
│   ├── metrics_table.csv                # Métricas em CSV
│   └── relatorio.html                   # Relatório interativo
```

## 📈 Métricas Reportadas

### Métricas Básicas
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coeficiente de determinação

### Métricas Normalizadas
- **EQMN1**: Erro Quadrático Médio Normalizado pela Variância
  - Fórmula: MSE / Var(y_true)
  - Interpretação: < 0.1 excelente, < 0.5 bom
- **EQMN2**: Erro Quadrático Médio Normalizado pelo Modelo Naive
  - Fórmula: MSE / MSE_naive
  - Interpretação: < 1.0 melhor que naive, < 0.5 excelente

## 🎨 Relatório HTML Interativo

Gere um relatório demonstrativo interativo:
```bash
python generate_interactive_report.py
```

**Funcionalidades do relatório**:
- 📊 Métricas detalhadas (R², RMSE, MAE, MSE, MAPE, EQMN1, EQMN2)
- 🖼️ Visualização de imagens em tela cheia
- 📈 Gráficos organizados por modelo
- 📋 Comparações interativas
- 👨‍🎓 Informações do autor

## ⚙️ Personalização

> **📖 Para configurações avançadas, consulte [README.md](README.md#configurações)**

### Modificar Configurações da Série
Edite `config/config.py`:
```python
MACKEY_GLASS_CONFIG = {
    'n_points': 5000,    # Menos pontos para teste rápido
    'tau': 20,
    'gamma': 0.2,
    'beta': 0.4,
    'n': 18,
    'x0': 0.8
}
```

### Modificar Parâmetros de Treinamento
```python
TRAINING_CONFIG = {
    'epochs': 50,        # Menos épocas para teste
    'learning_rate': 5e-4,
    'patience': 10,      # Early stopping mais agressivo
    # ...
}
```

### Modificar Dataset
```python
DATASET_CONFIG = {
    'window_size': 10,   # Janela menor
    'batch_size': 4096,  # Batch menor se pouca memória
    'train_ratio': 0.8,  # Mais dados para validação
    # ...
}
```

## 🔍 Troubleshooting

### Erro de CUDA
O código detecta automaticamente GPU/CPU. Se houver problemas:
```python
# Em config/config.py
DEVICE = 'cpu'  # Forçar uso de CPU
```

### Erro de Memória
Reduza o batch size:
```python
DATASET_CONFIG = {
    'batch_size': 1024,  # Reduzir de 8192
    # ...
}
```

### Dependências em falta
```bash
pip install torch numpy matplotlib seaborn tqdm scipy scikit-learn pandas
```

### Erro de Path/Import
Certifique-se de estar no diretório correto:
```bash
cd mackey_glass_prediction/experiments
python run_experiment.py
```

## 💡 Dicas de Uso

### Para Comparação Rápida
```bash
# Modelos pequenos para teste
python run_experiment.py --models mlp_small lstm_small gru_small --no-save
```

### Para Experimento Completo
```bash
# Todos os modelos (pode demorar várias horas)
python run_experiment.py --models all --output-dir experimento_completo
```

### Para Modelo Específico com Análise
```bash
# Um modelo com resultados salvos
python run_experiment.py --models lstm_medium
```

### Para Reproduzir Resultados
O código usa seeds fixas (RANDOM_SEED = 42) para reprodutibilidade. 
Para resultados diferentes, modifique em `config/config.py`:
```python
RANDOM_SEED = 123  # Ou qualquer outro valor
```

## 🎯 Cenários Comuns

### Desenvolvimento/Debug
```bash
python run_experiment.py --models mlp_small --no-save
```

### Comparação Científica
```bash
python run_experiment.py  # Modelos principais
```

### Análise Completa
```bash
python run_experiment.py --models all
```

### Modelo para Produção
```bash
python run_experiment.py --models gru_medium  # Bom equilíbrio
```

## 📊 Interpretação dos Resultados

### Arquivo CSV
```csv
Model,MSE,RMSE,MAE,MAPE,R²,EQMN1,EQMN2,Training_Time,Parameters
mlp_large,0.000234,0.015301,0.011234,1.89,0.9912,0.0456,0.2341,67.3,59073
```

### Console Output
```
LSTM_MEDIUM concluído com sucesso!
Tempo de treinamento: 78.5 segundos
Melhor loss de validação: 0.000068
Épocas treinadas: 31
Parâmetros: 50,113

MÉTRICAS DE VALIDAÇÃO:
MSE: 0.000068
RMSE: 0.008281
MAE: 0.006564
MAPE: 0.72%
R²: 0.9987
EQMN1: 0.0425
EQMN2: 0.1234
```

## 📚 Links Úteis

- **[📖 README.md](README.md)** - Documentação completa e configurações avançadas
- **[📊 RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md)** - Visão estratégica do projeto
- **[📈 RESULTADOS_FINAIS.md](RESULTADOS_FINAIS.md)** - Análise dos resultados experimentais

---
**Desenvolvido por**: Rafael Ratacheski de Sousa Raulino  
**Disciplina**: Redes Neurais Profundas - UFG 2025/1