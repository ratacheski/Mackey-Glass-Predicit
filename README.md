# 🧠 Trabalho 2 - Redes Neurais Profundas
**Predição de Séries Temporais Mackey-Glass com Arquiteturas MLP, LSTM e GRU**

---

## 📋 Navegação

- [📖 README](README.md) ← **Você está aqui**
- [🚀 Como Usar](COMO_USAR.md)
- [📊 Resumo Executivo](RESUMO_EXECUTIVO.md)
- [📈 Resultados Finais](RESULTADOS_FINAIS.md)

## 🌐 Demo Online

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://ratacheski.github.io/Mackey-Glass-Predicit/)

🔴 **Acesse o relatório interativo online**: [Clique aqui para ver o demo](https://ratacheski.github.io/Mackey-Glass-Predicit/relatorio.html)

## 🎯 Objetivo

Este projeto implementa e compara **três arquiteturas de redes neurais** (MLP, LSTM, GRU) para predição da série temporal Mackey-Glass, explorando diferentes configurações e técnicas avançadas para análise abrangente de desempenho.

# Predição de Séries Temporais Mackey-Glass com Redes Neurais

Este projeto implementa três tipos de redes neurais (MLP, LSTM, GRU) para predição de séries temporais de Mackey-Glass usando PyTorch, com múltiplas configurações e variações para cada modelo.

## 📚 Navegação

- **[📖 README.md](README.md)** - Documentação completa (você está aqui)
- **[🚀 COMO_USAR.md](COMO_USAR.md)** - Guia rápido de uso
- **[📊 RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md)** - Visão geral do projeto
- **[📈 RESULTADOS_FINAIS.md](RESULTADOS_FINAIS.md)** - Resultados dos experimentos

---

## Estrutura do Projeto

```
mackey_glass_prediction/
├── config/
│   └── config.py                    # Configurações centralizadas dos experimentos
├── data/
│   └── mackey_glass_generator.py    # Geração da série e datasets
├── experiments/
│   └── run_experiment.py            # Script principal para executar experimentos
├── models/
│   ├── __init__.py                  # Importação dos modelos
│   ├── mlp_model.py                 # Modelo MLP com variações
│   ├── lstm_model.py                # Modelo LSTM com variações
│   └── gru_model.py                 # Modelo GRU com variações
├── utils/
│   ├── training.py                  # Funções de treinamento e avaliação
│   └── visualization/               # Módulo completo de visualização
│       ├── __init__.py              # Funções básicas de plotting
│       ├── basic_plots.py           # Gráficos básicos
│       ├── comparison_plots.py      # Gráficos de comparação
│       ├── distribution_analysis.py # Análise de distribuição
│       ├── interactive_html.py      # Relatórios HTML interativos
│       ├── reports.py               # Geração de relatórios
│       ├── statistical_tests.py    # Testes estatísticos
│       └── utils.py                 # Utilitários de visualização
├── generate_interactive_report.py   # Script para gerar relatórios interativos
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Este arquivo
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

> **💡 Para uso rápido, consulte [COMO_USAR.md](COMO_USAR.md)**

### Executar experimentos principais (três modelos grandes)

```bash
cd experiments
python run_experiment.py
```

Por padrão, executa os modelos principais: `mlp_large`, `lstm_large`, `gru_large`

### Executar modelos específicos

```bash
# Executar apenas um modelo
python run_experiment.py --models mlp_medium

# Executar múltiplos modelos específicos
python run_experiment.py --models lstm_medium gru_medium mlp_small

# Executar todos os modelos disponíveis
python run_experiment.py --models all
```

### Opções de linha de comando

- `--models`: Especifica quais modelos executar
  - `main`: Executa os três modelos principais (padrão): `mlp_large`, `lstm_large`, `gru_large`
  - `all`: Executa todas as configurações disponíveis (15 modelos)
  - Nomes específicos: qualquer combinação dos modelos disponíveis
- `--no-save`: Não salva os resultados (útil para testes rápidos)
- `--output-dir`: Prefixo personalizado para a pasta de saída

### Exemplo de uso programático

```python
from experiments.run_experiment import run_single_experiment

# Executar um único experimento
results = run_single_experiment('lstm_medium')

# Acessar métricas
print(f"RMSE: {results['metrics']['RMSE']:.6f}")
print(f"R²: {results['metrics']['R²']:.6f}")
print(f"EQMN1: {results['metrics']['EQMN1']:.6f}")
print(f"EQMN2: {results['metrics']['EQMN2']:.6f}")
```

## Modelos Disponíveis

### MLP (Multi-Layer Perceptron)
- **mlp_small**: 2 camadas ocultas [64, 32], ~3K parâmetros
- **mlp_medium**: 3 camadas ocultas [128, 64, 32], ~14K parâmetros  
- **mlp_large**: 4 camadas ocultas [256, 128, 64, 32], ~59K parâmetros

### LSTM (Long Short-Term Memory)
- **lstm_small**: 1 camada, 32 unidades, ~8K parâmetros
- **lstm_medium**: 2 camadas, 64 unidades, ~50K parâmetros
- **lstm_large**: 3 camadas, 128 unidades, ~200K parâmetros
- **lstm_bidirectional**: 2 camadas bidirecionais, 64 unidades
- **lstm_attention**: 2 camadas com mecanismo de atenção, 64 unidades

### GRU (Gated Recurrent Unit)
- **gru_small**: 1 camada, 32 unidades, ~6K parâmetros
- **gru_medium**: 2 camadas, 64 unidades, ~37K parâmetros
- **gru_large**: 3 camadas, 128 unidades, ~150K parâmetros
- **gru_bidirectional**: 2 camadas bidirecionais, 64 unidades
- **gru_attention**: 2 camadas com mecanismo de atenção, 64 unidades

## Configurações

As configurações estão centralizadas em `config/config.py`:

### Parâmetros da Série Mackey-Glass
```python
MACKEY_GLASS_CONFIG = {
    'n_points': 10000,  # Número de pontos
    'tau': 20,          # Parâmetro de delay
    'gamma': 0.2,       # Parâmetro gamma
    'beta': 0.4,        # Parâmetro beta
    'n': 18,            # Parâmetro n
    'x0': 0.8           # Valor inicial
}
```

### Configurações do Dataset
```python
DATASET_CONFIG = {
    'window_size': 20,        # Tamanho da janela de entrada
    'prediction_steps': 1,    # Passos à frente para predizer
    'train_ratio': 0.9,       # 90% para treino, 10% para validação
    'batch_size': 8192,       # Tamanho do batch
    'shuffle_train': True     # Embaralhar dados de treino
}
```

### Configurações de Treinamento
```python
TRAINING_CONFIG = {
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 15,           # Early stopping
    'min_delta': 1e-6,        # Mínima melhoria
    'use_scheduler': True,    # Scheduler de learning rate
    'save_best_model': True,
    'save_final_model': True
}
```

## Resultados e Visualizações

Após executar os experimentos, são gerados:

### 1. Estrutura de Saída
```
experiments/results/
├── final_report_[timestamp]/
│   ├── 01_visao_geral_[timestamp].png       # Visão geral comparativa
│   ├── [modelo]/                            # Pasta para cada modelo
│   │   ├── training_curves.png              # Curvas de treinamento
│   │   ├── predictions.png                  # Gráfico de predições
│   │   ├── best_model.pth                   # Melhor modelo salvo
│   │   └── final_model.pth                  # Modelo final
│   ├── 99_tabela_metricas_[timestamp].png   # Tabela de métricas
│   ├── 99_comparacao_metricas_[timestamp].png # Comparação visual
│   ├── metrics_table.csv                    # Métricas em CSV
│   └── relatorio.html                       # Relatório HTML interativo
```

### 2. Relatório HTML Interativo
```bash
# Gerar relatório interativo demonstrativo
python generate_interactive_report.py
```

### 3. Métricas Avaliadas

#### Métricas Básicas
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coeficiente de determinação

#### Métricas Normalizadas
- **EQMN1**: Erro Quadrático Médio Normalizado pela Variância
  - **Fórmula**: EQMN1 = MSE / Var(y_true)
  - **Interpretação**: Valores menores indicam melhor performance. < 0.1 é excelente, < 0.5 é bom
- **EQMN2**: Erro Quadrático Médio Normalizado pelo Modelo Naive
  - **Fórmula**: EQMN2 = MSE / MSE_naive (modelo de persistência)
  - **Interpretação**: < 1.0 indica que o modelo é melhor que naive, < 0.5 é excelente

### 4. Visualizações Geradas
- Gráficos de loss durante treinamento
- Comparação entre predições e valores reais
- Análise de distribuição de resíduos
- Gráficos Q-Q e CDF dos resíduos
- Comparação visual entre modelos
- Tabelas de métricas formatadas

## Série Temporal Mackey-Glass

A série é gerada pela equação diferencial com atraso:
```
dx/dt = βx(t-τ)/(1 + x(t-τ)^n) - γx(t)
```

Parâmetros padrão:
- τ = 20 (delay)
- β = 0.4
- γ = 0.2
- n = 18
- x₀ = 0.8 (condição inicial)

## Desenvolvimento

### Adicionando novos modelos

1. Crie um novo arquivo em `models/` seguindo o padrão dos existentes
2. Implemente a classe herdando de `torch.nn.Module`
3. Adicione métodos obrigatórios: `forward()`, `get_model_info()`, `print_model_summary()`
4. Atualize `models/__init__.py`
5. Adicione configuração em `config/config.py` no dicionário `MODEL_CONFIGS`

### Exemplo de novo modelo
```python
# models/transformer_model.py
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super().__init__()
        # Implementação do modelo
        
    def forward(self, x):
        # Forward pass
        
    def get_model_info(self):
        """Retornar informações do modelo para relatórios"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Transformer',
            # Adicione outras informações específicas do modelo
        }
        
    def print_model_summary(self):
        """Imprimir resumo do modelo"""
        info = self.get_model_info()
        print(f"Modelo: {info['architecture']}")
        print(f"Parâmetros totais: {info['total_parameters']:,}")
        print(f"Parâmetros treináveis: {info['trainable_parameters']:,}")
```

### Personalizando Configurações

Para criar experimentos personalizados, edite `config/config.py`:

```python
# Adicionar nova configuração de modelo
MODEL_CONFIGS['meu_modelo_custom'] = {
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

## Dependências Principais

- **PyTorch**: Framework de deep learning
- **NumPy**: Computação numérica
- **Matplotlib**: Visualização básica
- **Seaborn**: Visualização estatística
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Métricas e utilities
- **SciPy**: Computação científica
- **TQDM**: Barras de progresso

## Links Úteis

- **[🚀 Guia de Uso Rápido](COMO_USAR.md)** - Como executar os experimentos
- **[📊 Resumo Executivo](RESUMO_EXECUTIVO.md)** - Visão geral do projeto
- **[📈 Resultados Finais](RESULTADOS_FINAIS.md)** - Análise dos resultados experimentais

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Autor

**Rafael Ratacheski de Sousa Raulino**  
Mestrando em Engenharia Elétrica e de Computação - UFG  
Disciplina: Redes Neurais Profundas - 2025/1 