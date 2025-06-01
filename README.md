# Predição de Séries Temporais Mackey-Glass com Redes Neurais

Este projeto implementa três tipos de redes neurais (MLP, LSTM, GRU) para predição de séries temporais de Mackey-Glass usando PyTorch.

## Estrutura do Projeto

```
mackey_glass_prediction/
├── config/
│   └── config.py              # Configurações dos experimentos
├── data/
│   └── mackey_glass_generator.py  # Geração da série e datasets
├── experiments/
│   └── run_experiment.py      # Script principal para executar experimentos
├── models/
│   ├── __init__.py           # Importação dos modelos
│   ├── mlp.py               # Modelo MLP
│   ├── lstm.py              # Modelo LSTM
│   └── gru.py               # Modelo GRU
├── utils/
│   ├── training.py          # Funções de treinamento e avaliação
│   └── visualization.py     # Funções de visualização
├── requirements.txt         # Dependências do projeto
└── README.md               # Este arquivo
```

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd mackey_glass_prediction
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Executar experimentos principais (MLP, LSTM, GRU)

```bash
cd experiments
python run_experiment.py
```

### Executar modelos específicos

```bash
# Executar apenas MLP
python run_experiment.py --models mlp

# Executar LSTM e GRU
python run_experiment.py --models lstm gru

# Executar todos os modelos disponíveis
python run_experiment.py --models all
```

### Opções de linha de comando

- `--models`: Especifica quais modelos executar
  - `main`: Executa os três modelos principais (padrão)
  - `all`: Executa todas as configurações disponíveis
  - Nomes específicos: `mlp`, `lstm`, `gru`, etc.
- `--no-save`: Não salva os resultados (útil para testes)

### Exemplo de uso programático

```python
from experiments.run_experiment import run_single_experiment

# Executar um único experimento
results = run_single_experiment('mlp')

# Acessar métricas
print(f"RMSE: {results['metrics']['RMSE']:.6f}")
print(f"R²: {results['metrics']['R²']:.6f}")
```

## Modelos Implementados

### 1. MLP (Multi-Layer Perceptron)
- Rede neural totalmente conectada
- Configurável com múltiplas camadas ocultas
- Dropout para regularização
- Diferentes funções de ativação

### 2. LSTM (Long Short-Term Memory)
- Rede neural recorrente com mecanismo de memória
- Suporte para múltiplas camadas
- Opção de bidirecionalidade
- Mecanismo de atenção opcional

### 3. GRU (Gated Recurrent Unit)
- Versão simplificada do LSTM
- Menos parâmetros que LSTM
- Suporte para múltiplas camadas
- Opção de bidirecionalidade

## Configurações

As configurações dos experimentos estão em `config/config.py`. Você pode ajustar:

- **Série Mackey-Glass**: tau, gamma, beta, número de pontos
- **Dataset**: tamanho da janela, divisão treino/teste, batch size
- **Treinamento**: épocas, learning rate, early stopping
- **Modelos**: arquitetura específica para cada tipo

### Exemplo de modificação de configuração

```python
# Em config/config.py
MLP_CONFIG = {
    'model_type': 'mlp',
    'input_size': 20,      # Tamanho da janela
    'hidden_sizes': [128, 64, 32],  # Camadas ocultas
    'output_size': 1,
    'dropout_rate': 0.2,
    'activation': 'relu'
}
```

## Resultados

Após executar os experimentos, são gerados:

1. **Relatório HTML**: Comparação visual entre modelos
2. **Gráficos de loss**: Curvas de treinamento e validação
3. **Métricas detalhadas**: MSE, RMSE, MAE, MAPE, R²
4. **Predições sequenciais**: Visualização das predições futuras
5. **Dados salvos**: Modelos treinados e resultados em formato pickle

### Estrutura dos resultados

```
results/
├── final_report_[timestamp]/
│   ├── 01_visao_geral_[timestamp].png
│   ├── [modelo]/
│   │   ├── training_history.png
│   │   └── predictions.png
│   ├── metrics_comparison.png
│   ├── metrics_table.csv
│   └── relatorio.html
```

## Métricas Avaliadas

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coeficiente de determinação

## Série Temporal Mackey-Glass

A série é gerada pela equação diferencial:
```
dx/dt = βx(t-τ)/(1 + x(t-τ)^n) - γx(t)
```

Parâmetros padrão:
- τ = 17 (delay)
- β = 0.2
- γ = 0.1
- n = 10
- x₀ = 1.2 (condição inicial)

## Desenvolvimento

### Adicionando novos modelos

1. Crie um novo arquivo em `models/`
2. Implemente a classe herdando de `torch.nn.Module`
3. Adicione métodos `forward()`, `get_model_info()` e `print_model_summary()`
4. Atualize `models/__init__.py`
5. Adicione configuração em `config/config.py`

### Exemplo de novo modelo

```python
# models/transformer.py
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super().__init__()
        # Implementação do modelo
        
    def forward(self, x):
        # Forward pass
        
    def get_model_info(self):
        # Retornar informações do modelo
        
    def print_model_summary(self):
        # Imprimir resumo do modelo
```

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes. 