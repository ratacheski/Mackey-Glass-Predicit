# Como Usar o Projeto - Guia Rápido

## 🚀 Início Rápido

### 1. Instalar Dependências
```bash
cd mackey_glass_prediction
pip install -r requirements.txt
```

### 2. Executar Experimentos

#### Todos os modelos principais (recomendado)
```bash
cd experiments
python run_experiment.py
```

#### Modelo específico
```bash
# Apenas MLP
python run_experiment.py --models mlp_medium

# Apenas LSTM
python run_experiment.py --models lstm_medium

# Apenas GRU
python run_experiment.py --models gru_medium

# Múltiplos modelos
python run_experiment.py --models mlp_medium lstm_medium
```

#### Teste rápido (sem salvar)
```bash
python run_experiment.py --models mlp_medium --no-save
```

## 📊 Resultados

Após executar, os resultados ficam em:
- `experiments/results/final_report_<timestamp>/`
- Gráficos de comparação
- Tabelas de métricas
- Modelos treinados

## 🔧 Personalização

### Modificar Configurações
Edite `config/config.py` para ajustar:
- Parâmetros da série Mackey-Glass
- Arquitetura dos modelos
- Hiperparâmetros de treinamento

### Exemplo de Modificação
```python
# Em config/config.py
MACKEY_GLASS_CONFIG = {
    'n_points': 5000,  # Menos pontos para teste rápido
    'tau': 17,
    'gamma': 0.1,
    'beta': 0.2,
    'n': 10,
    'x0': 1.2
}
```

## 📁 Estrutura dos Arquivos

```
mackey_glass_prediction/
├── config/config.py          # Configurações
├── data/                     # Geração de dados
├── models/                   # Definições dos modelos
├── utils/                    # Funções auxiliares
├── experiments/              # Scripts de execução
├── requirements.txt         # Dependências
└── README.md               # Documentação completa
```

## 🎯 Modelos Disponíveis

- `mlp_medium`: MLP com 3 camadas ocultas
- `lstm_medium`: LSTM com 2 camadas
- `gru_medium`: GRU com 2 camadas

## 📈 Métricas Avaliadas

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coeficiente de determinação

## 🔍 Troubleshooting

### Erro de CUDA
Se não tiver GPU, o código automaticamente usa CPU.

### Erro de Memória
Reduza o `batch_size` em `config/config.py`:
```python
DATASET_CONFIG = {
    'batch_size': 16,  # Reduzir de 32 para 16
    # ...
}
```

### Erro de Dependências
```bash
pip install torch numpy matplotlib seaborn tqdm scipy scikit-learn pandas
```

## 💡 Dicas

1. **Teste rápido**: Use `--no-save` para não gerar arquivos
2. **Comparação**: Execute todos os modelos para ver a comparação completa
3. **Personalização**: Modifique `config/config.py` para seus experimentos